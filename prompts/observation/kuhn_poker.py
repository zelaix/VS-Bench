import numpy as np

from models.base_model import ObservationPrompt

rules = (
    "1. Kuhn poker is a two-player card game. The deck includes only three cards: King (K) > Queen (Q) > Jack (J).\n"
    "2. At the start of each game, both player_0 and player_1 place 1 chip into the pot as a blind ante.\n"
    "3. Each player is dealt a private card, and the third card is set aside unseen.\n"
    "4. The two players take turns acting, starting with player_0. A player can choose to:\n"
    "    a. <PASS>: place no additional chips into the pot.\n"
    "    b. <BET>: place 1 additional chip into the pot.\n"
    "5. If a player chooses to <PASS> after the other player's <BET>, the betting player wins the pot.\n"
    "6. If both players choose to <PASS> or both players choose to <BET>, the player with the higher card wins the pot."
)


def get_observation_prompt(observation):
    obs = observation.obs
    agent_id = observation.agent_id
    image_paths = observation.image_paths
    legal_actions = ", ".join(list(observation.legal_actions.values()))
    obs_string = obs_to_string(obs, len(image_paths) > 0)

    text = (f"GAME RULES:\n{rules}\n\n"
            f"PLAYER INFORMATION:\nYou are player {agent_id}.\n\n"
            f"GAME HISTORY:\n{obs_string}\n\n"
            f"LEGAL ACTIONS:\n{legal_actions}.")
    observation_prompt = ObservationPrompt(text=text, image_paths=image_paths)

    return observation_prompt


def obs_to_string(obs, with_image):
    history = ["1. Blind ante: both player_0 and player_1 place 1 chip into the pot."]
    if with_image:
        history.append("2. Deal: the cards and chips are shown in the image.")
    else:
        deck = ["Jack (J)", "Queen (Q)", "King (K)"]
        card = deck[np.argmax(obs[2:5])]
        history.append(f"2. Deal: your card is {card}.")

    action_set = ["<PASS>", "<BET>"]
    num_turns = int(np.sum(obs[5:]))
    for i in range(num_turns):
        player_id = i % 2
        action = action_set[np.argmax(obs[5 + 2 * i:7 + 2 * i])]
        history.append(f"{i + 3}. Turn {i + 1}: player {player_id} chooses to {action}.")

    obs_string = "\n".join(history)
    return obs_string
