from textwrap import dedent


def get_action_prompt(observation):
    mark = 'X' if observation.agent_id == 0 else 'O'
    action_prompt = dedent(f"""\
        INSTRUCTIONS:
        Now it is your turn to choose an action. You should output your action in the following JSON format:
        ```json
        {{
            "action": "{mark}(i,j)"
        }}
        ```
        where i is the row index and j is the column index.""")
    return action_prompt
