from prompts.system import atari_pong
from prompts.system import battle_of_the_colors
from prompts.system import breakthrough
from prompts.system import coin_dilemma
from prompts.system import hanabi
from prompts.system import kuhn_poker
from prompts.system import monster_hunt
from prompts.system import overcooked
from prompts.system import tic_tac_toe

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


def get_system_prompt(game_name):
    return mapping[game_name].get_system_prompt()
