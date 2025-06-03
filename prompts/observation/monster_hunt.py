from models.base_model import ObservationPrompt


def get_observation_prompt(observation):
    image_paths = observation.image_paths
    current_role = 'red' if observation.agent_id == 0 else 'blue'
    rules_visual = (
        "1. Monster Hunt is a general-sum game played on a 5x5 grid board with two players (red and blue), one monster, and two apples.\n"
        "2. The monster moves towards the closest player in each step.\n"
        "3. Players move in the grid-world and receive rewards on different events:\n"
        "    a. One player eats an apple: the player +2 points and the apple respawns at a random position.\n"
        "    b. One player encounters the monster alone: the player -2 points and respawns at a random position.\n"
        "    c. Two players defeat the monster together: both players +5 points and the monster respawns at a random position."
    )
    rules_language = (
        "1. Monster Hunt is a general-sum game played on a 5x5 grid board with two players (red and blue), one monster, and two apples. "
        "The grid board is 0-indexed, with [0, 0] representing the bottom-left corner and [4, 4] representing the top-right corner.\n"
        "2. The monster moves towards the closest player in each step.\n"
        "3. Players move in the grid-world and receive rewards on different events:\n"
        "    a. One player eats an apple: the player +2 points and the apple respawns at a random position.\n"
        "    b. One player encounters the monster alone: the player -2 points and respawns at a random position.\n"
        "    c. Two players defeat the monster together: both players +5 points and the monster respawns at a random position."
    )
    information_visual_obs = (
        f"1. You are the {current_role} player.\n"
        "2. The current game frame and a table of events and counters are shown in the image.\n"
        "3. The red and blue players are represented by a red and blue pacman icon, respectively. "
        "The monster is represented by a black demon icon, and the apples are represented by green apple icons. "
        "If both players are in the same position, they are represented by a half-red-half-blue pacman icon.")
    information_no_image = (f"1. You are the {current_role} player.\n"
                            f"2. Your position is: {observation.obs[0:2]}.\n"
                            f"3. The other player's position is: {observation.obs[2:4]}.\n"
                            f"4. The monster's position is: {observation.obs[4:6]}.\n"
                            f"5. The apples' positions are: {observation.obs[6:8]} and {observation.obs[8:10]}.\n")
    control = ("1. <UP>: move one step upward.\n"
               "2. <DOWN>: move one step downward.\n"
               "3. <LEFT>: move one step left.\n"
               "4. <RIGHT>: move one step right.\n"
               "5. <STAY>: stay in the current position.")
    text = (f"GAME RULES:\n{rules_visual if image_paths else rules_language }\n\n"
            f"PLAYER INFORMATION:\n{information_visual_obs if image_paths else information_no_image}\n\n"
            f"LEGAL ACTIONS:\n{control}")
    observation_prompt = ObservationPrompt(text=text, image_paths=image_paths)
    return observation_prompt
