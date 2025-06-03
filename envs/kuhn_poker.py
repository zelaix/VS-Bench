from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os
import pyspiel

from envs.base_env import BaseEnv, Observation
from utils.recorder import Recorder


class KuhnPoker(BaseEnv, env_type="kuhn_poker"):
    """Kuhn Poker game environment using OpenSpiel."""

    def __init__(
        self,
        visual_obs=True,
        image_dir=None,
        recording_type='gif',
        recording_fps=2,
    ):
        self._env = pyspiel.load_game("kuhn_poker")
        self.state = None
        self.num_agents = self._env.num_players()
        self.image_paths = []

        self.visual_obs = visual_obs
        if self.visual_obs:
            assert image_dir is not None, "image_dir must not be None."
            self.image_dir = image_dir
            self.recorders = [
                Recorder(image_dir, recording_type, recording_fps, f"recording_agent_{i}")
                for i in range(self.num_agents)
            ]

    @property
    def current_player(self):
        return self.state.current_player()

    def reset(self, seed=0):
        """seed
            0: J, Q
            1: J, K
            2: Q, K
            3: Q, J
            4: K, J
            5: K, Q
        """
        card_0 = (seed // 2) % 3
        card_1 = ((seed - 3) // 2) % 3
        self.state = self._env.new_initial_state()
        self.state.apply_action(card_0)
        self.state.apply_action(card_1)
        self.bets = [1, 1]
        if self.visual_obs:
            for recorder in self.recorders:
                recorder.clear()
            self.image_paths = self._save_image()
        return [self._get_observation(i) for i in range(self.num_agents)]

    def step(self, actions):
        if self.state.is_terminal():
            raise RuntimeError("Cannot apply action on a terminal state.")

        agent_action = actions[self.current_player]
        self.bets[self.current_player] += agent_action  # 0: <PASS>, 1: <BET>
        self.state.apply_action(agent_action)
        if self.visual_obs:
            self.image_paths = self._save_image()
            if self.state.is_terminal():
                for recorder in self.recorders:
                    recorder.save()

        observations = [self._get_observation(i) for i in range(self.num_agents)]
        rewards = self.state.rewards()
        dones = [self.state.is_terminal()] * self.num_agents
        info = self._get_info()

        return observations, rewards, dones, info

    def _get_observation(self, agent_id):
        """Return the Observation object for agent agent_id.
            * obs[0:2]: agent id
            * obs[2:5]: one-hot encode of self card (J, Q, K)
            * obs[5:11]: one-hot encode of actions (pass, bet)
        """
        if agent_id == self.current_player:
            return Observation(
                obs=np.array(self.state.information_state_tensor(agent_id)),
                agent_id=agent_id,
                image_paths=self.image_paths,
                legal_actions=self._get_legal_actions(agent_id),
                serialized_state=self.state.serialize(),
                regex_patterns=self.regex_patterns,
                addition_info=None,
            )
        else:
            return None

    @property
    def regex_patterns(self):
        patterns = [
            (r'```json\s*\{\s*"action"\s*:\s*"([^"]+)"\s*\}\s*```', lambda m: m.strip()),
            (r'"action"\s*:\s*"([^"]+)"', lambda m: m.strip()),
            (r'```json\s*\{\s*"card"\s*:\s*"([^"]+)"\s*\}\s*```', lambda m: m.strip()),
            (r'"card"\s*:\s*"([^"]+)"', lambda m: m.strip()),
            (r'<\s*pass\s*>', lambda _: '<PASS>'),
            (r'<\s*bet\s*>', lambda _: '<BET>'),
            (r'\$\\boxed\{\\text\{(PASS|BET)\}\}\$', lambda m: f"<{m.upper()}>"),
            (r'\b(pass|bet)\b', lambda m: f"<{m.upper()}>"),
        ]
        return patterns

    def _get_info(self):
        if len(self.state.history()) == 3:
            deck = ["Jack (J)", "Queen (Q)", "King (K)"]
            card_0 = deck[self.state.history()[0]]
            card_1 = deck[self.state.history()[1]]
            return {"cards": [card_0, card_1]}
        elif self.state.is_terminal():
            returns = self.state.returns()
            winner = int(np.argmax(returns))
            return {
                "returns": returns,
                "winner": winner,
            }
        else:
            return None

    def _get_legal_actions(self, agent_id):
        legal_actions = dict()
        actions = self.state.legal_actions(agent_id)
        for a in actions:
            legal_actions[a] = self._action_to_string(agent_id, a)
        return legal_actions

    def _action_to_string(self, agent_id, action):
        if action == 0:
            return "<PASS>"
        else:
            return "<BET>"

    def _save_image(self, font="Monaco"):
        # image parameters
        img_width, img_height = 800, 800
        x_player, x_card, x_chips = 120, 350, 650
        y_player_header, y_player_0, y_player_1 = 40, 80, 440

        card_height = 300
        chip_height = 200
        y_player_0_center = y_player_0 + card_height // 2
        y_player_1_center = y_player_1 + card_height // 2

        # cards
        deck = ["J", "Q", "K"]
        card_0 = deck[self.state.history()[0]] if self.current_player == 0 else "unknown"
        card_1 = deck[self.state.history()[1]] if self.current_player == 1 else "unknown"
        card_0 = self._load_card_image(card_0, card_height)
        card_1 = self._load_card_image(card_1, card_height)

        # chips in pot
        chip_0 = self._load_chip_image(self.bets[0], chip_height)
        chip_1 = self._load_chip_image(self.bets[1], chip_height)

        # draw image
        canvas = Image.new("RGB", (img_width, img_height), color=(0, 127, 0))
        draw = ImageDraw.Draw(canvas)
        # player labels
        player_font = ImageFont.truetype(f"./images/kuhn_poker/{font}.ttf", 30)
        draw.text((x_player, y_player_0_center), "player_0", font=player_font, fill="white", anchor="mm")
        draw.text((x_player, y_player_1_center), "player_1", font=player_font, fill="white", anchor="mm")
        # headers
        header_font = ImageFont.truetype(f"./images/kuhn_poker/{font}.ttf", 30)
        draw.text((x_card, y_player_header), "Private Card", font=header_font, fill="white", anchor="mm")
        draw.text((x_chips, y_player_header), "Chips in Pot", font=header_font, fill="white", anchor="mm")
        # cards
        canvas.paste(card_0, (x_card - 107, y_player_0), card_0)
        canvas.paste(card_1, (x_card - 107, y_player_1), card_1)
        # chips
        canvas.paste(chip_0, (x_chips - 100, y_player_0_center - 100), chip_0)
        canvas.paste(chip_1, (x_chips - 100, y_player_1_center - 100), chip_1)

        # save image
        num_steps = self.state.move_number() - 2
        image_path = os.path.join(self.image_dir, f"step_{num_steps}.png")
        canvas.save(image_path)
        if self.current_player in range(self.num_agents):
            self.recorders[self.current_player].add_frame(image_path)
        return [image_path]

    def _load_card_image(self, card, card_height):
        if card not in ["J", "Q", "K", "unknown"]:
            raise ValueError(f"Illegal card: {card}")
        work_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        img = Image.open(f"{work_dir}/images/kuhn_poker/{card}.png").convert("RGBA")
        card_width = int(img.width / img.height * card_height)
        return img.resize((card_width, card_height), Image.LANCZOS)

    def _load_chip_image(self, chip, chip_height):
        if chip not in [1, 2]:
            raise ValueError(f"Illegal chip: {chip}")
        work_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        img = Image.open(f"{work_dir}/images/kuhn_poker/{chip}.png").convert("RGBA")
        chip_width = int(img.width / img.height * chip_height)
        return img.resize((chip_width, chip_height), Image.LANCZOS)
