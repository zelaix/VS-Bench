import os
import re

from models.base_model import ObservationPrompt
from utils.helper import image_to_b64


def get_observation_prompt(observation):
    image_paths = observation.image_paths

    rules = (
        "1. Overcooked is a cooperative game where two chefs collaborate to cook and serve soups in 50 timesteps.\n"
        "2. The chefs can move in the available area and cannot move to the counter.\n"
        "3. The chefs can interact with the object on the tile that they are facing.\n"
        "4. A soup is cooked in the following steps:\n"
        "    a. Pick up (interact) 1 onion and place (interact) it in the pot.\n"
        "    b. After placing 3 onions in the pot, open (interact) the pot and cook for 5 timesteps. The pot will show how long the soup has been cooked.\n"
        "    c. When the pot shows the number 5, the soup is finished. Pick up (interact) a dish to plate (interact) the soup.\n"
        "    d. Deliver the soup and put (interact) it on the serving location.")

    legal_actions = ("1. <UP>: face up and move up one tile if possible.\n"
                     "2. <DOWN>: face down and move down one tile if possible.\n"
                     "3. <RIGHT>: face right and move right one tile if possible.\n"
                     "4. <LEFT>: face left and move left one tile if possible.\n"
                     "5. <STAY>: stay in the current tile and do nothing.\n"
                     "6. <INTERACT>: interact with the object on the tile that you are facing.")

    if len(image_paths) > 0:
        recent_image_paths = get_recent_frames(os.path.dirname(image_paths[0]), max_frames=4)

    else:
        recent_image_paths = []

    information = obs_to_string(observation, len(recent_image_paths) > 0, recent_image_paths)
    action_history = action_history_to_string(observation, len(recent_image_paths) > 0, recent_image_paths)
    text = (f"GAME RULES:\n{rules}\n\n"
            f"PLAYER INFORMATION:\n{information}\n\n"
            f"HISTORY ACTIONS:\n{action_history if action_history else ''}\n\n"
            f"LEGAL ACTIONS:\n{legal_actions}")

    observation_prompt = ObservationPrompt(text=text, image_paths=recent_image_paths)
    return observation_prompt


def action_history_to_string(observation, visual_obs, recent_image_paths=None):
    if visual_obs:
        action_history = ""
        if hasattr(observation, 'addition_info') and observation.addition_info:
            action_history = f"{observation.addition_info}"
        return action_history
    else:
        return ''


def obs_to_string(observation, visual_obs, recent_image_paths=None):
    if visual_obs:
        # Description when image sequence is available
        hat_color = 'blue' if observation.agent_id == 0 else 'green'

        current_state = [image_to_b64(path) for path in recent_image_paths]
        player_hold = extract_player_hold_info(observation.serialized_state, observation.agent_id)
        hold_text = f"{player_hold}"

        obs_string = (
            f"1. You are controlling chef_{observation.agent_id} in the {hat_color} hat.\n"
            f"2. You are holding {hold_text} currently.\n"
            f"3. The image sequence shows the {len(current_state)} most recent game frames, with the last image being the current game frame. Each image shows the frame and object legend, with the timestep in the top left corner."
        )
    else:
        current_player = observation.agent_id
        obs_string = (
            "The size of the room is a 5x4 grid (X and Y), the X-axis runs from left to right, the Y-axis runs from top to bottom. The init overall layout is: "
            "XXPXX"
            "OMMMO"
            "XMMMX"
            "XDXSX"
            "The letter X stands for table, P for cooking station, O and o stand for onions, D and d for dishs, S for service desk, and M for empty area which is available for chefs to move. For example, onions local in (0,1) (4,1). "
            "When the onion or dish is on the table or being held by chef, a lowercase o or d will be added after its corresponding character. "
            "When the onion is placed on the cooking station, it will be denoted as P{øøø, P{øøø means that there are three onions on the cooking station. "
            "The numbers 0 and 1 represent the chef, and the direction arrow ↑ ↓ ← → represents the direction the chef is facing. Each object occupies a grid size, and the chef moves one grid distance at a time. "
            "And when the cooking station cooks the soup, it will show how long it has been cooked, such as P{øøø1 means that it has been cooked in 1 time steps. P{øøø✓ means that the soup is finished. 0{øøø✓ means that the chef 0 is holding a dish of soup. "
            f"You are controlling chef {current_player}. "
            f"The current detailed state is shown below: \n{observation.serialized_state}")

    return obs_string


def get_recent_frames(image_parent_path, max_frames=1):
    if not image_parent_path:
        return []

    all_files = os.listdir(image_parent_path)

    sorted_files = sorted(all_files, key=lambda x: int(os.path.splitext(x)[0][5:]))
    recent_files = sorted_files[-max_frames:]
    return [os.path.join(image_parent_path, fname) for fname in recent_files]


def extract_player_hold_info(serialized_state, agent_id):

    lines = serialized_state.strip().split('\n')

    player_info = ""
    for line in lines:
        if f"Chef {agent_id} local in" in line:
            hold_info_match = re.search(r'hold\s+(.*?)\.?$', line)
            if hold_info_match:
                player_info = hold_info_match.group(1)
                break

    return player_info
