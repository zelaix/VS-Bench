from textwrap import dedent


def get_action_prompt(observation):
    action_prompt = dedent(f"""\
        INSTRUCTIONS:
        Now you should choose an action base on the game state in the current game frame. You should output your action in the following JSON format:
        ```json
        {{
            "action": "<ACTION>"
        }}
        ```
        where <ACTION> is one of <UP>, <DOWN>, <LEFT>, <RIGHT>, <STAY>, <INTERACT>.""")
    return action_prompt
