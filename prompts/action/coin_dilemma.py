from textwrap import dedent


def get_action_prompt(observation):
    legal_actions = ", ".join(list(observation.legal_actions.values()))
    action_prompt = dedent(f"""\
        INSTRUCTIONS:
        You should output your action in the following JSON format:
        ```json
        {{
            "action": "<ACTION>"
        }}
        ```
        where <ACTION> is one of the following: {legal_actions}.""")
    return action_prompt
