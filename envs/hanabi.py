from matplotlib import patches, gridspec
from collections import deque
import matplotlib.pyplot as plt
import os
import pyspiel
import random
import re
import textwrap

from envs.base_env import BaseEnv, Observation
from utils.recorder import Recorder


class Hanabi(BaseEnv, env_type="hanabi"):

    def __init__(
        self,
        players=2,
        colors=5,
        ranks=5,
        hand_size=5,
        max_information_tokens=8,
        max_life_tokens=3,
        visual_obs=True,
        image_dir=None,
        recording_type='gif',
        recording_fps=2,
    ):
        self.num_agents = players
        if self.num_agents != 2:
            raise ValueError(f"Curren Hanabi only support 2 players, with current player num of {self.num_agents}")

        self.game_parameters = {
            "players": players,
            "colors": colors,
            "ranks": ranks,
            "hand_size": hand_size,
            "max_information_tokens": max_information_tokens,
            "max_life_tokens": max_life_tokens,
        }

        self.visual_obs = visual_obs
        if self.visual_obs:
            assert image_dir is not None, "image_dir must not be None."
            self.image_dir = image_dir
            self.recorders = [
                Recorder(image_dir, recording_type, recording_fps, f"recording_agent_{i}")
                for i in range(self.num_agents)
            ]

        self.history_size = 4
        self.history = deque(maxlen=self.history_size)
        self.state = None
        self.image_paths = []

    @property
    def current_player(self):
        return self.state.current_player()

    def reset(self, seed=0):
        self.game_parameters["seed"] = seed
        self._env = pyspiel.load_game("hanabi", self.game_parameters)
        self.state = self._env.new_initial_state()
        self.num_steps = 0

        while self.state.is_chance_node():
            outcomes_with_probs = self.state.chance_outcomes()
            actions, probs = zip(*outcomes_with_probs)
            action = random.choices(actions, weights=probs)[0]
            self.state.apply_action(action)

        if self.visual_obs:
            for recorder in self.recorders:
                recorder.clear()
            self.image_paths = self._save_image()
        return [self._get_observation(i) for i in range(self.num_agents)]

    def step(self, actions):
        if self.state.is_terminal():
            raise RuntimeError("Cannot apply action on a terminal state.")

        agent_action = actions[self.current_player]
        self.history.append(
            f"player {self.current_player} select {self._action_to_string(self.current_player, agent_action)}")
        self.state.apply_action(agent_action)
        self.num_steps += 1
        chance_node_action = self._handle_chance_node()

        if self.visual_obs:
            self.image_paths = self._save_image()
            if self.state.is_terminal():
                for recorder in self.recorders:
                    recorder.save()

        observations = [self._get_observation(i) for i in range(self.num_agents)]
        rewards = self.state.rewards()
        dones = [self.state.is_terminal()] * self.num_agents
        info = self._get_info(chance_node_action)
        return observations, rewards, dones, info

    def _handle_chance_node(self):
        if self.state.is_chance_node():
            outcomes_with_probs = self.state.chance_outcomes()
            actions, probs = zip(*outcomes_with_probs)
            action = random.choices(actions, weights=probs)[0]
            chance_node_action = self._action_to_string(self.current_player, action)
            self.state.apply_action(action)
            if self.state.is_chance_node():
                raise ValueError(f"the state is still a chance node\n{self.state}")
            return chance_node_action
        return None

    def _get_observation(self, agent_id):
        if agent_id == self.current_player:
            return Observation(
                obs={
                    'obs': self._get_player_obs(),
                    'config': self.game_parameters,
                    'history': self.history
                },
                agent_id=agent_id,
                image_paths=self.image_paths,
                legal_actions=self._get_legal_actions(agent_id),
                serialized_state=self.state.serialize(),
                regex_patterns=self.regex_patterns,
                addition_info=None,
            )
        else:
            return None

    def _get_player_obs(self):
        obs = self.state.observation_string()
        lines = obs.strip().split('\n')
        result = {
            "life_tokens": None,
            "info_tokens": None,
            "fireworks": "",
            "deck_size": None,
            "discards": "",
            "current_player_hand": [],
            "other_hands": [],
            "current_player_info": {},
            "other_info": {},
        }
        in_hands = False
        hands_section = []
        current_player_index = None
        index = None

        for line in lines:
            if line.startswith("Life tokens:"):
                result["life_tokens"] = int(line.split(":")[1].strip())
            elif line.startswith("Info tokens:"):
                result["info_tokens"] = int(line.split(":")[1].strip())
            elif line.startswith("Fireworks:"):
                result["fireworks"] = line.split(":", 1)[1].strip()
            elif line.strip() == "Hands:":
                in_hands = True
            elif line.strip() == "Cur player":
                current_player_index = len(hands_section)
            elif line.startswith("Deck size:"):
                result["deck_size"] = int(line.split(":")[1].strip())
                in_hands = False
            elif line.startswith("Discards:"):
                result["discards"] = line.split(":", 1)[1].strip()
            elif in_hands:
                if line.strip() != "-----":
                    hands_section.append(line.strip())
                else:
                    index = len(hands_section)

        if current_player_index == 0:
            result["other_hands"] = hands_section[index:]
            result["current_player_hand"] = hands_section[:index]
        else:
            result["other_hands"] = hands_section[:index]
            result["current_player_hand"] = hands_section[index:]

        for i, card in enumerate(result['current_player_hand']):
            card_info = card.split("|")[-1]
            letters = re.findall(r"[A-Za-z]", card_info)
            digits = re.findall(r"\d", card_info)
            result['current_player_info'][f'{i}'] = {'digits': digits, 'colors': letters}

        for i, card in enumerate(result['other_hands']):
            visible_card = card.split('||')[0].strip()
            card_info = card.split("|")[-1]
            letters = re.findall(r"[A-Za-z]", card_info)
            digits = re.findall(r"\d", card_info)
            result['other_info'][f'{i}'] = {'visible_card': visible_card, 'digits': digits, 'colors': letters}

        return result

    @property
    def regex_patterns(self):
        patterns = [(r'```json\s*\{\s*"action"\s*:\s*"([^"]+)"\s*\}\s*```', lambda m: m.strip()),
                    (r'"action"\s*:\s*"([^"]+)"', lambda m: m.strip()),
                    (r'Play\s+(\d+)', lambda m: f"(Play {m.strip()})"),
                    (r'Discard\s+(\d+)', lambda m: f"(Discard {m.strip()})"),
                    (r'Reveal player \+1 color\s+([A-Za-z])', lambda m: f"(Reveal player +1 color {m.strip()})"),
                    (r'Reveal player \+1 rank\s+(\d+)', lambda m: f"(Reveal player +1 rank {m.strip()})"),
                    (r'\$\boxed{\\text{([^"}]+)}}\$', lambda m: m.strip()),
                    (r'\$\boxed{{\\text{([^"}]+)}}}\$', lambda m: m.strip()), (r'\([^()]+\)', lambda m: m.strip())]
        return patterns

    def _get_info(self, chance_node_action):
        if self.state.is_terminal():
            returns = self.state.returns()
            if chance_node_action:
                return {"returns": returns, "chance_node_action": chance_node_action}
            else:
                return {"returns": returns}
        else:
            if chance_node_action:
                return {"chance_node_action": chance_node_action}
            else:
                return None

    def _get_legal_actions(self, agent_id):
        legal_actions = dict()
        actions = self.state.legal_actions(agent_id)
        for a in actions:
            legal_actions[a] = self._action_to_string(agent_id, a)
        return legal_actions

    def _action_to_string(self, agent_id, action):
        return self.state.action_to_string(agent_id, action)

    def _save_image(self):
        if self.state.is_terminal():
            return None

        obs = self._get_player_obs()
        COLOR = dict(R="#c62828", Y="#f9a825", G="#388e3c", W="#ffffff", B="#039be5")
        BORDER_COLOR = dict(R="#8e0000", Y="#c17900", G="#1b5e20", W="#bdbdbd", B="#006db3")
        n_you = len(obs["current_player_hand"])
        n_opp = len(obs["other_hands"])
        n_cols = max(n_you, n_opp, 4)

        card_w, card_h = 1.5, 5
        fig_w, fig_h = n_cols * 2.5, 13.0
        fig = plt.figure(figsize=(fig_w, fig_h))

        history_len = len(self.history)
        history_size = 0.8 * history_len if history_len > 0 else 0.8
        height_ratios = [0.8, 0.8, history_size, 0.7, card_h, 0.7, card_h, 1.0, 0.7, card_h, 1.0]
        gs = gridspec.GridSpec(11, n_cols, height_ratios=height_ratios)
        fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02, hspace=0.4, wspace=0.4)

        line0 = "   ".join([
            f"Life tokens: {obs['life_tokens']}", f"Info tokens: {obs['info_tokens']}", f"Deck size: {obs['deck_size']}"
        ])
        ax0 = fig.add_subplot(gs[0, :])
        ax0.axis("off")
        ax0.text(0.5, 0.5, line0, ha="center", va="center", fontsize=18, fontweight="bold")

        discards = obs["discards"] or "None"
        line1 = f"Discard pile: {discards}"
        ax1 = fig.add_subplot(gs[1, :])
        ax1.axis("off")
        wrapped1 = textwrap.fill(line1, width=93, subsequent_indent=' ' * 21)
        ax1.text(0, 0.5, wrapped1, ha="left", va="center", fontsize=16, fontweight="bold")

        hist = list(self.history)
        ax2 = fig.add_subplot(gs[2, :])
        ax2.axis("off")

        if not hist:
            ax2.text(0, 0.5, "Action history: None", ha="left", va="center", fontsize=16, fontweight="bold")
        else:
            lines = []
            for rev_idx, action in enumerate(reversed(hist[-self.history_size:])):
                if rev_idx == 0:
                    lines.append(f"Action history: 1 turn ago {action}")
                else:
                    lines.append(f"{' ' * 24} {rev_idx+1} turns ago: {action}")
            history_text = "\n".join(lines)
            ax2.text(0, 0.5, history_text, ha="left", va="center", fontsize=16, fontweight="bold")

        ax_ft = fig.add_subplot(gs[3, 0])
        ax_ft.axis("off")
        ax_ft.text(0, 0.5, "Fireworks:", ha="left", va="top", fontsize=20, fontweight="bold")

        fw = {p[0]: int(p[1:]) for p in obs["fireworks"].split()}
        for j, color in enumerate(["R", "Y", "G", "W", "B"]):
            if j == len(fw):
                break
            axf = fig.add_subplot(gs[4, j])
            axf.axis("off")
            rank = fw.get(color, 0)
            face = COLOR[color]
            edge = BORDER_COLOR[color]
            rect = patches.Rectangle((0, 0), card_w, card_h, ec=edge, fc=face, lw=10)
            axf.add_patch(rect)
            txtc = "black" if color == "W" else "white"
            axf.text(card_w / 2, card_h / 2, str(rank), ha="center", va="center", fontsize=40, color=txtc)
            axf.set_xlim(0, card_w)
            axf.set_ylim(0, card_h)

        ax_ty = fig.add_subplot(gs[5, 0])
        ax_ty.axis("off")
        ax_ty.text(0, 0.5, f"Player {self.current_player} (You)", ha="left", va="top", fontsize=20, fontweight="bold")
        for k in range(n_you):
            axf = fig.add_subplot(gs[6, k])
            axf.axis("off")
            rect = patches.Rectangle((0, 0), card_w, card_h, ec="#263238", fc="#9e9e9e", lw=10)
            axf.add_patch(rect)
            axf.text(card_w / 2, card_h / 2, "?", ha="center", va="center", fontsize=40, color="white")
            axf.set_xlim(0, card_w)
            axf.set_ylim(0, card_h)
            axi = fig.add_subplot(gs[7, k])
            axi.axis("off")
            d = obs["current_player_info"][str(k)]
            cs = ", ".join(d["colors"]) or "—"
            rs = ", ".join(d["digits"]) or "—"
            axi.text(0, 0.5, f"Card {k}:\nColor: {cs}\nRank: {rs}", ha="left", va="center", fontsize=14,
                     fontweight="bold", wrap=True)

        ax_to = fig.add_subplot(gs[8, 0])
        ax_to.axis("off")
        ax_to.text(0, 0.5, f"Player {1 - self.current_player}", ha="left", va="top", fontsize=20, fontweight="bold")
        for k in range(n_opp):
            axf = fig.add_subplot(gs[9, k])
            axf.axis("off")
            rep = obs["other_hands"][k]
            ccode = re.findall(r"[A-Za-z]", rep)[0]
            rdigit = re.findall(r"\d+", rep)[0]
            face = COLOR.get(ccode, "#cccccc")
            edge = BORDER_COLOR[ccode]
            rect = patches.Rectangle((0, 0), card_w, card_h, ec=edge, fc=face, lw=10)
            axf.add_patch(rect)
            txtc = "black" if ccode == "W" else "white"
            axf.text(card_w / 2, card_h / 2, rdigit, ha="center", va="center", fontsize=40, color=txtc)
            axf.set_xlim(0, card_w)
            axf.set_ylim(0, card_h)
        for k in range(n_opp):
            axi = fig.add_subplot(gs[10, k])
            axi.axis("off")
            d = obs["other_info"][str(k)]
            cs = ", ".join(d["colors"]) or "—"
            rs = ", ".join(d["digits"]) or "—"
            axi.text(0, 0.5, f"Card {k}:\nColor: {cs}\nRank: {rs}", ha="left", va="center", fontsize=14,
                     fontweight="bold", wrap=True)

        image_path = os.path.join(self.image_dir, f"step_{self.num_steps}.png")
        fig.savefig(image_path, dpi=200)
        plt.close(fig)
        if self.current_player in range(self.num_agents):
            self.recorders[self.current_player].add_frame(image_path)
        return [image_path]
