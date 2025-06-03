from models.base_model import ObservationPrompt

color_set = ['R(Red)', 'Y(Yellow)', 'G(Green)', 'W(White)', 'B(Blue)']


def get_rule(config):
    c, r = config["colors"], config["ranks"]
    per_color = 3 + 2 * (r - 2) + 1
    total_cards = c * per_color

    color_names = ', '.join(color_set[:c])

    rank_range_str = (
        f"Each color contains {per_color} cards: three of rank 1, two each of rank 2 through {r-1}, and one of rank {r}, "
        if r - 1 > 2 else f"Each color contains {per_color} cards: three of rank 1, two of rank 2, and one of rank 3, ")

    rules = (
        f"1. Hanabi is a cooperative card game for {config['players']} players.\n"
        f"2. The deck consists of {c} colors: {color_names}, with ranks ranging from 1 to {r}. "
        f"{rank_range_str}"
        f"for a total of {total_cards} cards.\n"
        f"3. Each player holds {config['hand_size']} cards in hand.\n"
        f"4. There are {config['max_information_tokens']} Info tokens (used to give hints) "
        f"and {config['max_life_tokens']} Life tokens (penalties for misplays).\n"
        "5. As in blind man's bluff, players can see each other's cards but they cannot see their own. "
        "Play proceeds around the table; each turn, a player must take one of the following actions:\n"
        "    a. (Play i): play the i-th card from your hand (0-indexed) and attempt to add it to the cards already played. "
        "This is successful if the card is a 1 in a suit that has not yet been played, or if it is the next number sequentially in a suit that has been played. "
        "Otherwise a Life token is consumed and the misplayed card is discarded. "
        f"Successfully playing a {r} of any suit replenishes one Info token. Whether the play was successful or not, the player draws a replacement card from the deck (if any remain).\n"
        "    b. (Discard i): discard the i-th card from your hand and draw a replacement card from the deck (if any remain). "
        "The discarded card is out of the game and can no longer be played. Discarding a card replenishes one Info token.\n"
        "    c. (Reveal player +1 color c): spend one Info token to reveal all cards of color c in the other player's hand.\n"
        "    d. (Reveal player +1 rank r): spend one Info token to reveal all cards of rank r in the other player's hand.\n"
        f"6. The game ends immediately when either all Life tokens are used up, resulting in a game loss with a score of 0, "
        f"or when all {r}s have been successfully played, resulting in a game win with a score of {r*c}. "
        "Otherwise, the game continues until the deck runs out and one final round is completed. "
        f"At the end of the game, the final score is calculated as the sum of the highest card played in each suit, up to a maximum of {r*c} points."
    )
    return rules


def get_observation_prompt(observation):
    obs = observation.obs['obs']
    config = observation.obs['config']
    history = observation.obs['history']
    agent_id = observation.agent_id
    image_paths = observation.image_paths
    legal_actions = ", ".join(list(observation.legal_actions.values()))

    obs_string = obs_to_string(obs, history, len(image_paths) > 0)

    text = (f"GAME RULES:\n{get_rule(config)}\n\n"
            f"PLAYER INFORMATION:\nYou are player {agent_id}.\n\n"
            f"GAME STATE:\n{obs_string}\n\n"
            f"LEGAL ACTIONS:\n{legal_actions}.")
    return ObservationPrompt(text=text, image_paths=image_paths)


def obs_to_string(obs_dict, history, visual_obs):
    if visual_obs:
        obs_str = []
        obs_str.append(
            "Below is a visual representation of the current game state:\n"
            "    - The first section, located above the image, presents the game's basic state information.\n"
            "    - The second section summarizes the most recent player actions.\n"
            "    - The third section displays the current firework stacks, with each color labeled by the highest successfully played rank.\n"
            "    - The fourth section shows your own hand, represented as gray squares marked with '?', reflecting the fact that you cannot see your own cards.\n"
            "    - The fifth section presents the other player's hand, with each card shown in its true color and rank, since it is fully visible to you.\n"
            "Below each card, you will find two lines of inferred information:\n"
            "    - Color: a list of all possible colors deduced for that card so far.\n"
            "    - Rank: a list of all possible ranks deduced for that card so far.\n"
            "The information displayed below your cards reflects the hints the other player has given you so far.\n"
            "The information below the other player's cards represents what they currently believe about their own cards, based on all the useful hints you have provided them up to this point. "
            "For example, below your first card you might see:\n"
            "    Card 0:\n"
            "    Color: R, Y\n"
            "    Rank: 2, 3\n"
            "indicating that your card 0 is either Red or Yellow and has rank 2 or 3.")

    else:
        discards = obs_dict['discards'] if obs_dict['discards'] else 'None'
        obs_str = [
            f"There are {obs_dict['life_tokens']} life tokens remaining.",
            f"There are {obs_dict['info_tokens']} information tokens remaining.",
            f"The current firework stacks are: {obs_dict['fireworks']}.",
            f"{obs_dict['deck_size']} cards remain in the draw pile.",
            f"The discard pile currently contains: {discards}.",
        ]
        obs_str.append(
            "Based on the hints you've received, you know each of your cards must be one of these colors and one of these ranks:"
        )
        for i, card_info in obs_dict['current_player_info'].items():
            colors = ', '.join(card_info['colors'])
            ranks = ', '.join(card_info['digits'])
            obs_str.append(f"    - Card {i}: one of the colors [{colors}] and one of the ranks [{ranks}].")

        obs_str.append("For the other player's visible cards, you see their face value as follows:")
        for i, card_info in obs_dict['other_info'].items():
            colors = ', '.join(card_info['colors'])
            ranks = ', '.join(card_info['digits'])
            visible_card = card_info['visible_card']
            obs_str.append(
                f"    - Card {i} (visible as {visible_card}): based on all hints so far, the other player believes it is one of the colors [{colors}] and one of the ranks [{ranks}]."
            )

    return "\n".join(obs_str)
