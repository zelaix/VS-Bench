from datetime import datetime
from typing import *
import dataclasses
import json
import numpy as np
import logging
import os
import time


@dataclasses.dataclass
class StepInfo:
    actions: List[int]
    agent_infos: List[Dict[str, Any]]  # None if the agent does not act
    rewards: List[float]
    dones: List[bool]
    env_info: Dict[int, str]


@dataclasses.dataclass
class PredictInfo:
    choose_action: int
    target_actions: List[int]
    agent_info: Dict[str, Any]
    reward: int


class Logger:
    """A logger that writes messages to both console and files (TXT and JSON)."""

    def __init__(self, config):
        self.start_time = time.time()
        self.exp_name = config["experiment"].get("name", "default")
        self.num_episodes = config["experiment"].get("num_episodes", 1)
        self.env_name = config["environment"]["type"]
        self.num_agents = len(config["agents"])
        self.returns = np.zeros((self.num_episodes, self.num_agents))
        self.steps = np.zeros(self.num_episodes, dtype=int)
        self.price_info = {
            f"episode_{i}": {
                "total_price": 0,
                "prompt_price": 0,
                "completion_price": 0,
            } for i in range(self.num_episodes)
        }
        self.results = {
            "config": config,
            "returns_mean": {f"agent_{i}": 0 for i in range(self.num_agents)},
            "returns_std": {f"agent_{i}": 0 for i in range(self.num_agents)},
            "steps_mean": 0,
            "steps_std": 0,
            "price_info": {
                "total_price": 0,
                "prompt_price": 0,
                "completion_price": 0,
            }
        }
        self.results.update({f"episode_{i}": dict() for i in range(self.num_episodes)})

        # log dir
        results_dir = config["experiment"].get("results_dir", "results")
        agents_name = []
        for agent_config in config["agents"]:
            type_ = agent_config["type"]
            model = agent_config["params"].get("model")
            if model is None:
                agents_name.append(type_)
            else:
                model_name = model["params"]["name"]
                agents_name.append(f"{type_}({model_name})")
        agents_name = "+".join(agents_name)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = os.path.join(results_dir, "decision-making", self.env_name, agents_name, self.exp_name,
                                    timestamp)
        self.log_file = os.path.join(self.log_dir, f"output.log")
        self.json_file = os.path.join(self.log_dir, f"results.json")
        os.makedirs(self.log_dir, exist_ok=True)

        # init logger
        logger_name = f"{self.env_name}_{agents_name}_{os.getpid()}_{timestamp}"
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        formatter = logging.Formatter("%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        self.logger.info(f"Text logging to {self.log_file}")
        self.logger.info(f"Json logging to {self.json_file}")
        self.logger.info(f"Environment: {self.env_name}")
        self.logger.info(f"Agents: {agents_name}")
        self.logger.info(f"Experiment: {self.exp_name}")

    # def log(self, message, level="info"):
    #     if hasattr(self.logger, level.lower()):
    #         getattr(self.logger, level.lower())(message)
    #     else:
    #         raise ValueError(f"Unknown log level: {level}.")

    def get_episode_dir(self, episode_id):
        episode_dir = os.path.join(self.log_dir, f"episode_{episode_id}")
        os.makedirs(episode_dir, exist_ok=True)
        return episode_dir

    def episode_start(self, episode_id):
        self.logger.info(f"===== Episode {episode_id} start =====")

    def step(self, episode_id, step_info):
        step = self.steps[episode_id]
        self.logger.info(f"step {step}: actions={step_info.actions}, env_info={step_info.env_info}")

        info = dict()
        for i, agent_info in enumerate(step_info.agent_infos):
            agent_info = agent_info or dict()  # if agent_info is None, initialize to empty dict
            agent_info["reward"] = step_info.rewards[i]
            agent_info["done"] = step_info.dones[i]
            info[f"agent_{i}"] = agent_info
        info["env_info"] = step_info.env_info
        self.results[f"episode_{episode_id}"][f"step_{step}"] = info

        for i, agent_info in enumerate(step_info.agent_infos):
            token_info = agent_info.get("token_info") if agent_info is not None else None
            if token_info is not None:
                self.price_info[f"episode_{episode_id}"]["total_price"] += token_info["total"]["total_price"]
                self.price_info[f"episode_{episode_id}"]["prompt_price"] += token_info["prompt"]["prompt_price"]
                self.price_info[f"episode_{episode_id}"]["completion_price"] += token_info["completion"][
                    "completion_price"]

        self.steps[episode_id] += 1

    def episode_end(self, episode_id, env_info):
        self.results[f"episode_{episode_id}"]["result"] = env_info
        self.returns[episode_id] = np.array(env_info["returns"])
        self.logger.info(f"done: returns = {env_info['returns']}")
        self.logger.info(f"===== Episode {episode_id} end =====")

    def save_results(self):
        for i in range(self.num_agents):
            self.results["returns_std"][f"agent_{i}"] = np.std(self.returns[:, i])
            self.results["returns_mean"][f"agent_{i}"] = np.mean(self.returns[:, i])
            self.results["steps_std"] = np.std(self.steps)
            self.results["steps_mean"] = np.mean(self.steps)

        for i in range(self.num_episodes):
            self.results["price_info"]["total_price"] += self.price_info[f"episode_{i}"]["total_price"]
            self.results["price_info"]["prompt_price"] += self.price_info[f"episode_{i}"]["prompt_price"]
            self.results["price_info"]["completion_price"] += self.price_info[f"episode_{i}"]["completion_price"]

        self.logger.info(f"num_episodes: {self.num_episodes}")
        self.logger.info(f"returns_mean: {self.results['returns_mean']}")
        self.logger.info(f"returns_std: {self.results['returns_std']}")
        self.logger.info(f"steps_mean: {self.results['steps_mean']}")
        self.logger.info(f"steps_std: {self.results['steps_std']}")

        results = convert_numpy_types(self.results)
        with open(self.json_file, 'w') as f:
            json.dump(results, f, indent=4)
        self.logger.info(f"Results saved to {self.json_file}")

    # def log_step(self, episode, step, step_info: StepInfo):
    #     info = dict()
    #     for i, agent_info in enumerate(step_info.agent_infos):
    #         agent_info = agent_info or dict()  # if agent_info is None, initialize to empty dict
    #         agent_info["reward"] = step_info.rewards[i]
    #         agent_info["done"] = step_info.dones[i]
    #         info[f"agent_{i+1}"] = agent_info
    #     info["env_info"] = step_info.env_info
    #     self.results[f"episode_{episode}"][f"step_{step}"] = info

    # def log_episode(self, episode, env_info):
    #     """
    #     Log the result of an episode and update results.

    #     Args:
    #         episode (int): Episode number.
    #         outcome (str): Outcome of the episode ('Player 0 wins', 'Player 1 wins', 'Draw').
    #         returns (list): Returns for each player.
    #         additional_info (dict, optional): Additional data to log for this episode.
    #     """
    #     episode_result = copy.deepcopy(env_info)
    #     env_info["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    #     agent1_token_info = {
    #         "step" : 0,
    #         "total_prompt_price" : 0,
    #         "average_prompt_price" : 0,
    #         "total_completion_price" : 0,
    #         "average_completion_price" : 0,
    #         "total_price" : 0,
    #         "average_price" : 0
    #     }
    #     agent2_token_info = {
    #         "step" : 0,
    #         "total_prompt_price" : 0,
    #         "average_prompt_price" : 0,
    #         "total_completion_price" : 0,
    #         "average_completion_price" : 0,
    #         "total_price" : 0,
    #         "average_price" : 0
    #     }

    #     episode_data = self.results.get(f"episode_{episode}", {})
    #     for i, return_value in enumerate(env_info["returns"]):
    #         self.results["average_returns"][f"agent_{i+1}"].append(return_value)
    #     episode_steps = len([k for k in episode_data.keys() if k.startswith("step_")])
    #     self.results["average_steps"].append(episode_steps)

    #     for step_str, step_data in episode_data.items():
    #         match = re.fullmatch(r"step_(\d+)", step_str)
    #         if not match:
    #             continue
    #         step = int(match.group(1))
    #         agent_key = "agent_1" if step % 2 == 0 else "agent_2"
    #         agent_info = agent1_token_info if step % 2 == 0 else agent2_token_info
    #         agent_step = step_data.get(agent_key, {})
    #         token_info = agent_step.get("token_info", {})
    #         if token_info:
    #             agent_info["step"] += 1
    #             agent_info["total_prompt_price"] += token_info.get("prompt", {}).get("prompt_price", 0)
    #             agent_info["total_completion_price"] += token_info.get("completion", {}).get("completion_price", 0)
    #     for agent_info in [agent1_token_info, agent2_token_info]:
    #         steps = agent_info["step"]
    #         if steps > 0:
    #             agent_info["average_prompt_price"] = agent_info["total_prompt_price"] / steps
    #             agent_info["average_completion_price"] = agent_info["total_completion_price"] / steps
    #             agent_info["total_price"] = agent_info["total_prompt_price"] + agent_info["total_completion_price"]
    #             agent_info["average_price"] = agent_info["average_prompt_price"] + agent_info["average_completion_price"]
    #     episode_result["agent1_token_info"] = agent1_token_info
    #     episode_result["agent2_token_info"] = agent2_token_info
    #     self.results[f"episode_{episode}"]["result"] = episode_result

    # def _save_results(self):
    #     """Save the aggregated results to both TXT and JSON files."""
    #     # Append summary to TXT file
    #     # summary = [
    #     #     f"Total games: {self.episodes}",
    #     #     f"Player 0 ({self.agent1_name}) wins: {self.results['wins']['Player0']} ({self.results['wins']['Player0'] / self.episodes * 100:.1f}%)",
    #     #     f"Player 1 ({self.agent2_name}) wins: {self.results['wins']['Player1']} ({self.results['wins']['Player1'] / self.episodes * 100:.1f}%)",
    #     #     f"Draws: {self.results['wins']['draws']} ({self.results['wins']['draws'] / self.episodes * 100:.1f}%)"
    #     # ]
    #     # for line in summary:
    #     #     self.log(line)

    #     # average returns
    #     for k, v in self.results["average_returns"].items():
    #         self.results["average_returns"][k] = np.mean(v)
    #         self.results["returns_std"][k] = np.std(v)
    #     # average steps
    #     self.results["steps_std"] = np.std(self.results["average_steps"])
    #     self.results["average_steps"] = np.mean(self.results["average_steps"])

    #     for episode_str, episode_data in self.results.items():
    #         if not episode_str.startswith("episode_"):
    #             continue

    #         for agent_key in ['agent1', 'agent2']:
    #             for part in ['prompt_', 'completion_', '']:
    #                 self.results["token_price_info"][agent_key][f"total_{part}price"] += self.results[episode_str]['result'][f'{agent_key}_token_info'][f"total_{part}price"]

    #     for agent_key in ['agent1', 'agent2']:
    #         for part in ['prompt_', 'completion_', '']:
    #             total = round(self.results["token_price_info"][agent_key][f"total_{part}price"], 6)
    #             average = round(total / self.episodes, 6)
    #             self.results["token_price_info"][agent_key][f"total_{part}price"] = total
    #             self.results["token_price_info"][agent_key][f"average_{part}price"] = average

    #     # Convert numpy types to native Python types
    #     results_to_save = convert_numpy_types(self.results)

    #     self.log(f"Total number of episodes: {self.episodes}")
    #     self.log(f"Average returns: {results_to_save['average_returns']}")
    #     self.log(f"Returns std: {results_to_save['returns_std']}")
    #     self.log(f"Average steps per episode: {results_to_save['average_steps']}")
    #     self.log(f"Steps std: {results_to_save['steps_std']}")

    #     # Save to JSON
    #     with open(self.json_file, 'w') as f:
    #         json.dump(results_to_save, f, indent=4)

    #     self.log(f"Results saved to {self.json_file}")

    def close(self):
        """Close all handlers to ensure proper cleanup."""
        end_time = time.time()
        self.logger.info(f"Experiment completed in {end_time - self.start_time:.2f}s.")

        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler)

    # def create_image_dir(self):
    #     """Create a dir for visual input for VLM"""
    #     new_folder_path = self.final_name
    #     os.makedirs(new_folder_path, exist_ok=True)
    #     return new_folder_path


class PredictLogger:

    def __init__(self, config, env_name, num_prediction):
        self.start_time = time.time()
        self.exp_name = config["experiment"].get("name", "default")
        self.num_prediction = num_prediction
        self.env_name = env_name
        self.price_info = {
            f"predict_{i}": {
                "total_price": 0,
                "prompt_price": 0,
                "completion_price": 0,
            } for i in range(self.num_prediction)
        }
        self.results = {
            "config": config,
            "num_predicts": self.num_prediction,
            "accuracy": 0,
            "price_info": {
                "total_price": 0,
                "prompt_price": 0,
                "completion_price": 0,
            }
        }
        self.results.update({f"predict_{i}": dict() for i in range(self.num_prediction)})

        results_dir = config["experiment"].get("results_dir", "results")
        agent_config = config["agent"]
        type_ = agent_config["type"]
        model = agent_config["params"].get("model")
        if model is None:
            agent_name = type_
        else:
            model_name = model["params"]["name"]
            agent_name = f"{type_}({model_name})"

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = os.path.join(results_dir, "strategic_reasoning", self.env_name, agent_name, self.exp_name,
                                    timestamp)
        self.log_file = os.path.join(self.log_dir, f"output.log")
        self.json_file = os.path.join(self.log_dir, f"results.json")
        os.makedirs(self.log_dir, exist_ok=True)

        # init logger
        logger_name = f"{self.env_name}_{agent_name}_{os.getpid()}_{timestamp}"
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        formatter = logging.Formatter("%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        self.logger.info(f"Text logging to {self.log_file}")
        self.logger.info(f"Json logging to {self.json_file}")
        self.logger.info(f"Environment: {self.env_name}")
        self.logger.info(f"Agent: {agent_name}")
        self.logger.info(f"Experiment: {self.exp_name}")

        # self.log_dir = log_dir
        # self.num_predicts = num_predicts
        # self.visual_obs = env_config['visual_obs']

        # self.game_name = env_config['game_name']
        # self.agent_name = agent_config['agent_config']['agent_type']
        # self.model_name = agent_config['model_config']['model_type']
        # self.results = {
        #     'game_config': env_config,
        #     "num_predicts": self.num_predicts,
        #     "accuracy" : 0,
        #     "token_price_info": {
        #         "total_prompt_price": 0,
        #         "total_completion_price": 0,
        #         "total_price": 0,
        #         "average_prompt_price": 0,
        #         "average_completion_price": 0,
        #         "average_price": 0
        #     },
        #     'visual_obs' : self.visual_obs,
        #     'agent_config': agent_config
        # }
        # # self.results.update({f"predict_{i}": {} for i in range(self.num_predicts)})

        # current_date = datetime.now().strftime("%Y-%m-%d")
        # current_time = datetime.now().strftime("%H-%M-%S")

        # save_dir = os.path.join(self.log_dir, "strategic_reasoning", self.game_name, current_date, current_time)
        # os.makedirs(save_dir, exist_ok=True)

        # suffix = 'visual' if self.visual_obs else 'non-visual'
        # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # self.base_name = f"{suffix}-{self.agent_name}({self.model_name})-{self.num_predicts}predicts_{timestamp}"
        # self.final_name = os.path.join(save_dir, self.base_name)
        # os.makedirs(self.final_name, exist_ok=True)
        # self.txt_file = os.path.join(self.final_name, f"log.log")
        # self.json_file = os.path.join(self.final_name, f"json.json")

        # unique_logger_name = f"{self.game_name}_{self.agent_name}({self.model_name})_{os.getpid()}_{int(time.time()*1000)}"
        # self.logger = logging.getLogger(unique_logger_name)
        # self.logger.setLevel(logging.INFO)
        # self.logger.propagate = False

        # formatter = logging.Formatter("%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        # file_handler = logging.FileHandler(self.txt_file)
        # file_handler.setLevel(logging.INFO)
        # file_handler.setFormatter(formatter)

        # console_handler = logging.StreamHandler()
        # console_handler.setLevel(logging.INFO)
        # console_handler.setFormatter(formatter)

        # if self.logger.hasHandlers():
        #     self.logger.handlers.clear()

        # self.logger.addHandler(file_handler)
        # self.logger.addHandler(console_handler)

        # self.logger.info(f"Logger initialized")
        # self.logger.info(f"Txt logging to {self.txt_file}")
        # self.logger.info(f"Json logging to {self.json_file}\n")

        # self.logger.info(
        #     f"===== Game {self.game_name} start " +
        #     f"Player: ({self.agent_name} ({self.model_name}))"
        # )

    # def log(self, message, level="info"):
    #     """
    #     Log a message to both console and TXT file.

    #     Args:
    #         message (str): Message to log.
    #         level (str): Log level ('info', 'warning', 'error').
    #     """
    #     if level.lower() == "info":
    #         self.logger.info(message)
    #     elif level.lower() == "warning":
    #         self.logger.warning(message)
    #     elif level.lower() == "error":
    #         self.logger.error(message)
    #     else:
    #         raise ValueError(f"Unknown log level: {level}")

    def log_predict(self, predict_id, predict_info: PredictInfo):
        self.results[f"predict_{predict_id}"] = {
            "choose_action": predict_info.choose_action,
            "target_actions": predict_info.target_actions,
            "reward": predict_info.reward,
            "agent_info": predict_info.agent_info,
        }
        self.results['accuracy'] += predict_info.reward
        token_info = predict_info.agent_info.get("token_info") if predict_info.agent_info is not None else None
        if token_info is not None:
            self.price_info[f"predict_{predict_id}"]["total_price"] += token_info["total"]["total_price"]
            self.price_info[f"predict_{predict_id}"]["prompt_price"] += token_info["prompt"]["prompt_price"]
            self.price_info[f"predict_{predict_id}"]["completion_price"] += token_info["completion"]["completion_price"]

        self.logger.info(f"===== Predict {predict_id} end. Reward = {predict_info.reward} =====")

    def save_results(self):
        for i in range(self.num_prediction):
            self.results["price_info"]["total_price"] += self.price_info[f"predict_{i}"]["total_price"]
            self.results["price_info"]["prompt_price"] += self.price_info[f"predict_{i}"]["prompt_price"]
            self.results["price_info"]["completion_price"] += self.price_info[f"predict_{i}"]["completion_price"]

        self.results["accuracy"] = (self.results["accuracy"] / self.num_prediction) * 100

        self.logger.info(f"num_prediction: {self.num_prediction}")
        self.logger.info(f"final accuracy: {self.results['accuracy']}")

        results = convert_numpy_types(self.results)
        with open(self.json_file, 'w') as f:
            json.dump(results, f, indent=4)
        self.logger.info(f"Results saved to {self.json_file}")

    def close(self):
        """Close all handlers to ensure proper cleanup."""
        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler)


def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj
