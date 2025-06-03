from envs.base_env import Observation
from models.base_model import ObservationPrompt
from prompts.observation import atari_pong
from prompts.observation import battle_of_the_colors
from prompts.observation import breakthrough
from prompts.observation import coin_dilemma
from prompts.observation import hanabi
from prompts.observation import kuhn_poker
from prompts.observation import monster_hunt
from prompts.observation import overcooked
from prompts.observation import tic_tac_toe

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


def get_observation_prompt(game_name, observation: Observation) -> ObservationPrompt:
    return mapping[game_name].get_observation_prompt(observation)
