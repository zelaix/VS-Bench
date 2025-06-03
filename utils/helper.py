from io import BytesIO
from PIL import Image
import base64
import json
import numpy as np
import os
import pandas as pd
import random
import yaml


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def load_config(exp):
    with open(f"configs/exp_configs/{exp}.yaml", 'r') as f:
        config = yaml.safe_load(f)

    env_name = config["environment"]
    with open(f"configs/env_configs/{env_name}.yaml", 'r') as f:
        config["environment"] = yaml.safe_load(f)
        config["environment"]["params"] = config["environment"].get("params", {})

    for agent_config in config["agents"]:
        agent_type = agent_config["type"]
        agent_params = {"env_name": env_name}
        with open(f"configs/agent_configs/{agent_type}.yaml", 'r') as f:
            default_config = yaml.safe_load(f)
        agent_params.update(default_config.get("params", {}))
        agent_params.update(agent_config.get("params", {}))
        if "model" in agent_params.keys():
            model_name = agent_params["model"]
            with open(f"configs/model_configs/{model_name}.yaml", 'r') as f:
                agent_params["model"] = yaml.safe_load(f)
        agent_config["params"] = agent_params

    return config


def load_reasoning_config(exp):
    with open(f"configs/exp_configs/reasoning.yaml", 'r') as f:
        config = yaml.safe_load(f)

    env_name = exp
    if env_name == 'tic_tac_toe':
        raise ValueError("Tic-tac-toe's reasoning datasets is not availble now")

    with open(f"configs/env_configs/{env_name}.yaml", 'r') as f:
        config["environment"] = yaml.safe_load(f)
        config["environment"]["params"] = config["environment"].get("params", {})

    agent_config = config["agent"]
    agent_type = agent_config["type"]
    agent_params = {"env_name": env_name}
    with open(f"configs/agent_configs/{agent_type}.yaml", 'r') as f:
        default_config = yaml.safe_load(f)
    agent_params.update(default_config.get("params", {}))
    agent_params.update(agent_config.get("params", {}))
    if "model" in agent_params.keys():
        model_name = agent_params["model"]
        with open(f"configs/model_configs/{model_name}.yaml", 'r') as f:
            agent_params["model"] = yaml.safe_load(f)
    agent_config["params"] = agent_params

    if agent_type == 'random_agent' or agent_config['params']['visual_obs']:
        dataset_path = os.path.join('data', f'{env_name}.parquet')
    else:
        raise ValueError("Language dataset is not supported")

    df = pd.read_parquet(dataset_path)
    df["next_legal_actions"] = df["next_legal_actions"].apply(json.loads)
    dataset = df.to_dict(orient="index")
    for _, dataset_i in dataset.items():
        dataset_i['next_action'] = dataset_i['next_action'].tolist()
    return config, dataset


def image_to_b64(image_path, reasoning=False):
    if not reasoning:
        # png -> base64
        image = Image.open(image_path)
        with BytesIO() as image_buffer:
            image.save(image_buffer, format="PNG")
            byte_data = image_buffer.getvalue()
            image_b64 = base64.b64encode(byte_data).decode("utf-8")
        return image_b64
    else:
        # base64 -> just return
        return image_path


def image_to_byte(image_path, reasoning=False):
    if reasoning:
        try:
            img_bytes = base64.b64decode(image_path)
        except Exception as e:
            print(f"Error decoding base64 image: {e}")
            return None
    else:
        with open(image_path, 'rb') as f:
            img_bytes = f.read()
    return img_bytes
