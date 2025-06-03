import numpy as np

from models.base_model import ObservationPrompt

rules = ("1. Tic-tac-toe is a two-player board game played on a three-by-three grid. "
         "The grid is 0-indexed, where (0,0) is the top-left corner and (2,2) is the bottom-right corner.\n"
         "2. Two players take turns placing their marks X and O in empty cells of the grid.\n"
         "3. The player who first places three of their marks in a horizontal, vertical, or diagonal line wins.\n"
         "4. If all cells are filled and no player wins, the game ends in a draw.")


def get_observation_prompt(observation):
    obs = observation.obs
    mark = 'X' if observation.agent_id == 0 else 'O'
    image_paths = observation.image_paths
    legal_actions = ", ".join(list(observation.legal_actions.values()))
    obs_string = obs_to_string(obs, len(image_paths) > 0)

    text = (f"GAME RULES:\n{rules}\n\n"
            f"PLAYER INFORMATION:\nYour mark is {mark}.\n\n"
            f"GAME STATE:\n{obs_string}\n\n"
            f"LEGAL ACTIONS:\n{legal_actions}.")
    observation_prompt = ObservationPrompt(text=text, image_paths=image_paths)

    return observation_prompt


def obs_to_string(obs, visual_obs):
    if visual_obs:
        obs_string = "The current grid is shown in the image."
    else:
        marks = ["#", "O", "X"]
        grid = "\n".join("".join(marks[v] for v in row) for row in np.argmax(obs, axis=0))
        obs_string = f"The current grid is shown below, where # represents an empty cell.\n{grid}"
    return obs_string
