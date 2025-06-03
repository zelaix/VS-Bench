from prompts.action import atari_pong
from prompts.action import battle_of_the_colors
from prompts.action import breakthrough
from prompts.action import coin_dilemma
from prompts.action import hanabi
from prompts.action import kuhn_poker
from prompts.action import monster_hunt
from prompts.action import overcooked
from prompts.action import tic_tac_toe

mapping = {
    'atari_pong': atari_pong,
    'atari_pong_multi': atari_pong,
    'battle_of_the_colors': battle_of_the_colors,
    'breakthrough': breakthrough,
    'coin_dilemma': coin_dilemma,
    'hanabi': hanabi,
    'kuhn_poker': kuhn_poker,
    'monster_hunt': monster_hunt,
    'overcooked': overcooked,
    'tic_tac_toe': tic_tac_toe,
}


def get_action_prompt(game_name, observation):
    return mapping[game_name].get_action_prompt(observation)
