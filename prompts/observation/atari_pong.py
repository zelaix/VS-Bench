from models.base_model import ObservationPrompt


def get_observation_prompt(observation):
    image_paths = observation.image_paths
    if not image_paths:
        raise ValueError("Atari Pong requires image input!")

    current_player = 'right' if observation.agent_id == 0 else 'left'
    rules = ("1. Atari Pong is a zero-sum game played on a 2D screen with two players (left and right) and a ball.\n"
             "2. Players each controls a paddle and receive rewards on different events:\n"
             "    a. If the ball passes your paddle: the opponent +1 point.\n"
             "    b. If the ball passes the opponent's paddle: you +1 point.\n"
             "3. The ball bounces off the top/bottom walls and the paddles.\n"
             "4. Paddles can only move vertically within the top and bottom walls.\n"
             "5. First player to score 3 points wins.")
    information = (
        f"1. You are controlling the {current_player} paddle.\n"
        f"2. The recent {len(image_paths)} game frames are given in chronological order, with the most recent frame at the end.\n"
        "3. The ball is represented by a white square, and the paddles are represented by vertical rectangles.\n"
        "4. Scores are displayed at the top of the screen.")
    control = ("1. <UP>: move paddle upward.\n"
               "2. <DOWN>: move paddle downward.\n"
               "3. <STAY>: maintain current position (paddle has momentum, it stops gradually).")
    if len(observation.legal_actions) == 4:
        control += "\n4. <FIRE>: serve the ball (only applicable after you score)."

    text = (f"GAME RULES:\n{rules}\n\n"
            f"PLAYER INFORMATION:\n{information}\n\n"
            f"LEGAL ACTIONS:\n{control}")
    observation_prompt = ObservationPrompt(text=text, image_paths=image_paths)
    return observation_prompt
