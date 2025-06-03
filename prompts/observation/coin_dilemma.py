from models.base_model import ObservationPrompt


def get_observation_prompt(observation):
    image_paths = observation.image_paths
    if not image_paths:
        raise ValueError("Coin Dilemma requires image input!")

    current_role = 'red' if observation.agent_id == 0 else 'blue'
    rules = (
        "1. The Coin Dilemma is a general-sum game played on a 5x5 grid board with two players (red and blue) and two types of coins (red and blue).\n"
        "2. Players receive rewards on different events:\n"
        "    a. A player collects one coin of its own color: the player +1 point.\n"
        "    b. A player collects one coin of the other player's color: the player +1 point, the other player -2 points.\n"
        "3. New coins spawn randomly on the board after each collection.")
    information = (
        f"1. You are the {current_role} player.\n"
        "2. The current game frame and a table of events and counters are shown in the image.\n"
        "3. The red and blue players are represented by a red and blue pacman icon, respectively. "
        "The red and blue coins are represented by red and blue coin icons, respectively. "
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
