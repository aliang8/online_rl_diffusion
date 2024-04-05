"""
Train policy with RL algorithm
"""

from absl import app, logging
import jax
import optax
import jax.numpy as jnp
import numpy as np
import haiku as hk
import os
import tqdm
import pickle
import time
import flax
import re
import collections
from ml_collections import ConfigDict, FieldReference, FrozenConfigDict, config_flags
from typing import Any
import pickle
from pathlib import Path
from ray import train, tune
from ray.train import RunConfig, ScalingConfig
from diffusion_rl.trainers.trainer import BaseRLTrainer
from diffusion_rl.trainers.vae_trainer import VAERLTrainer
from diffusion_rl.trainers.offline_trainer import OfflineTrainer
from diffusion_rl.trainers.generative_model_trainer import GenerativeModelTrainer

# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.01"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

_CONFIG = config_flags.DEFINE_config_file("config")

# shorthands for config parameters
psh = {
    "trainer": "t",
    # "entropy_weight": "ew",
    "seed": "s",
    # "policy_lr": "plr",
    # "gamma": "g",
    "env_id": "eid",
    "num_epochs": "ne",
    "num_iter_per_epoch": "iters",
    "num_trajs": "nt",
    "policy": {
        "name": "p",
        "n_timesteps": "nts",
        "beta_schedule": "bs",
        "time_embed_dim": "ted",
        "conditional": "cond",
    },
}


def update(source, overrides):
    """
    Update a nested dictionary or similar mapping.
    Modify ``source`` in place.
    """
    for key, value in overrides.items():
        if type(source[key]) != type(overrides[key]):
            source[key] = overrides[key]
        elif isinstance(value, collections.abc.Mapping) and value:
            returned = update(source.get(key, {}), value)
            source[key] = returned
        else:
            source[key] = overrides[key]
    return source


def train_model_fn(config):
    trial_dir = train.get_context().get_trial_dir()

    if trial_dir:
        # this is if we are running with Ray
        logging.info("trial dir: ", trial_dir)
        config["exp_dir"] = Path(trial_dir)
        base_name = Path(trial_dir).name
        config["exp_name"] = base_name
        # the group name is without seed
        config["group_name"] = re.sub("_s-\d", "", base_name)
        logging.info(f"wandb group name: {config['group_name']}")
    else:
        suffix = f"{config['exp_name']}_s-{config['seed']}_t-{config['trainer']}"
        config["exp_dir"] = Path(config["exp_dir"]) / "results" / suffix

    # wrap config in ConfigDict
    config = ConfigDict(config)

    print(config)
    if config.trainer == "gaussian":
        trainer_cls = BaseRLTrainer
    elif config.trainer == "vae":
        trainer_cls = VAERLTrainer
    elif config.trainer == "offline":
        trainer_cls = OfflineTrainer
    elif config.trainer == "generative":
        trainer_cls = GenerativeModelTrainer
    else:
        raise ValueError(f"Trainer {config.trainer} not implemented")

    trainer = trainer_cls(config)
    if config.mode == "train":
        trainer.train()
    elif config.mode == "eval":
        trainer.eval()


param_space = {
    # "env_id": tune.grid_search(
    #     [
    #         "halfcheetah-medium-v2",
    #         "halfcheetah-medium-expert-v2",
    #         "halfcheetah-medium-replay-v2",
    #     ]
    # )
    "num_trajs": tune.grid_search([5, 10, 25, 50, 100]),
    "policy": {
        "n_timesteps": tune.grid_search([100]),
        "beta_schedule": tune.grid_search(["cosine"]),
    },
}


def trial_str_creator(trial):
    trial_str = ""

    for k, override in param_space.items():
        if k in trial.config:
            if isinstance(override, dict) and "grid_search" not in override:
                for k2 in override.keys():
                    if k2 in trial.config[k]:
                        trial_str += f"{psh[k][k2]}-{trial.config[k][k2]}_"
            else:
                trial_str += f"{psh[k]}-{trial.config[k]}_"

    # also add keys to include
    for k, v in trial.config["keys_to_include"].items():
        if v is None:
            if k not in param_space:
                trial_str += f"{psh[k]}-{trial.config[k]}_"
        else:
            for k2 in v:
                if k not in param_space or (
                    k in param_space and k2 not in param_space[k]
                ):
                    trial_str += f"{psh[k][k2]}-{trial.config[k][k2]}_"

    trial_str = trial_str[:-1]
    print("trial_str: ", trial_str)
    return trial_str


def main(_):
    config = _CONFIG.value.to_dict()
    if config["smoke_test"] is False:
        config["use_wb"] = True
        # run with ray tune
        config = update(config, param_space)
        train_model = tune.with_resources(train_model_fn, {"cpu": 4, "gpu": 0.1})

        run_config = RunConfig(
            name=config["exp_name"],
            local_dir="/scr/aliang80/online_rl_diffusion/diffusion_rl/ray_results",
            storage_path="/scr/aliang80/online_rl_diffusion/diffusion_rl/ray_results",
            log_to_file=True,
        )
        tuner = tune.Tuner(
            train_model,
            param_space=config,
            run_config=run_config,
            tune_config=tune.TuneConfig(
                trial_name_creator=trial_str_creator,
                trial_dirname_creator=trial_str_creator,
            ),
        )
        results = tuner.fit()
        print(results)
    else:
        # run without ray tune
        train_model_fn(config)


if __name__ == "__main__":
    app.run(main)
