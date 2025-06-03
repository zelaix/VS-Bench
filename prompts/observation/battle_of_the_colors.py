from models.base_model import ObservationPrompt


def get_observation_prompt(observation):
    image_paths = observation.image_paths
    if not image_paths:
        raise ValueError("Battle of the Colors requires image input!")

    current_role = 'red' if observation.agent_id == 0 else 'blue'
    rules = (
        "1. The Battle of the Colors is a general-sum game played on a 5x5 grid board with two players (red and blue) and two types of blocks (red and blue).\n"
        "2. Players receive rewards on different events:\n"
        "    a. When both players are on a red block: red player +2 points, blue player +1 point, and the red block will be refreshed to a new random position.\n"
        "    b. When both players are on a blue block: red player +1 point, blue player +2 points, and the blue block will be refreshed to a new random position.\n"
        "    c. When players are on different blocks: both players +0 points, and both blocks will be refreshed to new random positions."
    )
    information = (
        f"1. You are the {current_role} player.\n"
        "2. The current game frame and a table of events and counters are shown in the image.\n"
        "3. The red and blue players are represented by red and blue pacman icons, respectively. "
        "The red and blue blocks are represented by red and blue rectangles, respectively. "
        "If both players are in the same position, they are represented by a half-red-half-blue pacman icon.")
    control = ("1. <UP>: move one step upward.\n"
               "2. <DOWN>: move one step downward.\n"
               "3. <LEFT>: move one step left.\n"
               "4. <RIGHT>: move one step right.\n"
               "5. <STAY>: stay in the current position.")
    text = (f"GAME RULES:\n{rules}\n\n"
            f"PLAYER INFORMATION:\n{information}\n\n"
            f"LEGAL ACTIONS:\n{control}")
    observation_prompt = ObservationPrompt(text=text, image_paths=image_paths)
    return observation_prompt
