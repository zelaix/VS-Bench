from textwrap import dedent


def get_action_prompt(observation):
    action_prompt = dedent(f"""\
        INSTRUCTIONS:
        Now it is your turn to choose an action. You should output your action in the following JSON format:
        ```json
        {{
            "action": "(ACTION)"
        }}
        ```
        where (ACTION) is one of the actions listed in the LEGAL ACTIONS section.""")
    return action_prompt
