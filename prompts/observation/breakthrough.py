import string

from models.base_model import ObservationPrompt


def get_rules(cols, rows):
    rules = (
        f"1. Breakthrough is a two-player strategy game played on a {rows}x{cols} grid.\n"
        f"2. Each player controls pieces of a color: 'White' or 'Black'. 'White' starts at the bottom (rows 1 and 2), while 'Black' starts at the top (rows {rows - 1} and {rows}).\n"
        f"3. If 'White' moves a piece to row {rows}, 'White' wins the game. Conversely, if 'Black' moves a piece to row 1, 'Black' wins the game.\n"
        "4. Players alternate turns, moving one piece per turn, with 'Black' going first.\n"
        "5. A piece may only move one space straight or diagonally forward, and only if the destination square is empty.\n"
        "6. A piece may only capture an opponent's piece by moving one space diagonally forward into its square. "
        "In this case, the opponent's piece is removed, and your piece takes its place.\n"
        "7. 'Black' moves forward by decreasing row indices (downward), while 'White' moves forward by increasing them (upward).\n"
        "8. Moves are specified by their start and end positions. For example, 'a2a3' indicates moving a piece from a2 (column a, row 2) to a3 (column a, row 3).\n"
        f"9. The board is labeled with columns a-{string.ascii_lowercase[cols - 1]} and rows 1-{rows}. "
        f"Thus, {string.ascii_lowercase[cols - 1]}{rows} is the top-right corner, and a1 is the bottom-left corner.")
    return rules


def get_observation_prompt(observation):
    obs = observation.obs['obs']
    cols = observation.obs['cols']
    rows = observation.obs['rows']
    mark = 'Black' if observation.agent_id == 0 else 'White'
    image_paths = observation.image_paths
    legal_actions = ", ".join(list(observation.legal_actions.values()))
    obs_string = obs_to_string(obs, len(image_paths) > 0)

    text = (f"GAME RULES:\n{get_rules(cols, rows)}\n\n"
            f"PLAYER INFORMATION:\nYour mark is {mark}.\n\n"
            f"GAME STATE:\n{obs_string}\n\n"
            f"LEGAL ACTIONS:\n{legal_actions}.")
    observation_prompt = ObservationPrompt(text=text, image_paths=image_paths)

    return observation_prompt


def obs_to_string(obs, visual_obs):
    if visual_obs:
        obs_string = "The current grid is shown in the image. Row labels are displayed on the left, while column labels appear at the bottom. The pieces are marked using their corresponding colors in the grid."
    else:
        obs_string = f"The current grid is shown below, where '.' represents an empty cell.\n{obs}Where 'b' refers to 'Black' and 'w' refers to 'White'."
    return obs_string
