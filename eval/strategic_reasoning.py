from copy import deepcopy
import asyncio

from agents.base_agent import BaseAgent
from envs.base_env import BaseEnv
from utils.helper import load_reasoning_config
from utils.logger import PredictLogger, PredictInfo


async def run_predict(predict_id, config, env, dataset_i, logger):
    agent = BaseAgent.from_config(config["agent"])
    choose_action, agent_info = await agent.predict(dataset_i, env.regex_patterns)
    target_actions = dataset_i['next_action']
    reward = 1 if choose_action in target_actions else 0
    predict_info = PredictInfo(choose_action, target_actions, agent_info, reward)
    logger.log_predict(predict_id, predict_info)
    # logger.log(f"===== Predict {index} end. Reward = {reward} =====")


async def run_strategic_reasoning(exp_name):
    config, dataset = load_reasoning_config(exp_name)
    logger = PredictLogger(config, exp_name, len(dataset))
    async_mode = config["experiment"].get("async_mode", True)
    config["environment"]["params"]["image_dir"] = ""
    env = BaseEnv.from_config(config["environment"])
    if async_mode:
        batch_size = config['experiment']['batch_size']
        for i in range(0, len(dataset), batch_size):
            batch_indices = range(i, min(i + batch_size, len(dataset)))
            tasks = [run_predict(i, deepcopy(config), env, dataset[i], logger) for i in batch_indices]
            await asyncio.gather(*tasks)
    else:
        for i in range(len(dataset)):
            await run_predict(i, deepcopy(config), env, dataset[i], logger)

    logger.save_results()
    logger.close()
