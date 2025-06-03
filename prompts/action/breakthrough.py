from textwrap import dedent
import string


def get_action_prompt(observation):
    rows = observation.obs['rows']
    cols = observation.obs['cols']
    action_prompt = dedent(f"""\
        INSTRUCTIONS:
        It is now your turn to select an action. Please output your move in the following JSON format:
        ```json
        {{
            "action": "xiyj"
        }}
        ```
        where:
        - "x" and "y" represent the column letters, ranging from 'a' to '{string.ascii_lowercase[cols-1]}'.
        - "i" and "j" represent the row numbers, ranging from 1 to {rows}.
        
        For example, "a2a3" means moving the piece from column 'a', row 2 to column 'a', row 3.""")
    return action_prompt
