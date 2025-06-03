from copy import deepcopy
import asyncio

from agents.base_agent import BaseAgent
from envs.base_env import BaseEnv
from utils.helper import load_config
from utils.logger import Logger, StepInfo


async def run_episode(episode_id, config, logger):
    seed = config["experiment"].get("seed", 0)
    config["environment"]["params"]["image_dir"] = logger.get_episode_dir(episode_id)
    env = BaseEnv.from_config(config["environment"])
    agents = [BaseAgent.from_config(cfg) for cfg in config["agents"]]

    logger.episode_start(episode_id)
    observations = env.reset(seed=seed + episode_id)
    done = False
    while not done:
        tasks = [agent.act(obs) for agent, obs in zip(agents, observations)]
        results = await asyncio.gather(*tasks)

        actions = [result[0] for result in results]
        agent_infos = [result[1] for result in results]

        observations, rewards, dones, env_info = env.step(actions)
        step_info = StepInfo(actions, agent_infos, rewards, dones, env_info)
        logger.step(episode_id, step_info)
        done = all(dones)
    logger.episode_end(episode_id, env_info)


async def run_decision_making(exp_name):
    config = load_config(exp_name)
    logger = Logger(config)

    async_mode = config["experiment"].get("async_mode", True)
    num_episodes = config["experiment"].get("num_episodes", 1)
    if async_mode:
        tasks = [run_episode(i, deepcopy(config), logger) for i in range(num_episodes)]
        await asyncio.gather(*tasks)
    else:
        for i in range(num_episodes):
            await run_episode(i, deepcopy(config), logger)

    logger.save_results()
    logger.close()
