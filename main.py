"""
Train policy with RL algorithm

Usage:
python3 main.py \
    --config=configs/rl_config.py \
    --config.mode=train \
"""

from absl import app
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
from ml_collections import ConfigDict, FieldReference, FrozenConfigDict, config_flags
from typing import Any
import mlogger
import pickle
from pathlib import Path
from ray import train, tune
from ray.train import RunConfig, ScalingConfig
from trainer import BaseRLTrainer
from vae_trainer import VAERLTrainer

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.01"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

_CONFIG = config_flags.DEFINE_config_file("config")


def train_model_fn(config):
    trial_dir = train.get_context().get_trial_dir()
    if trial_dir:
        print("Trial dir: ", trial_dir)
        config["root_dir"] = Path(trial_dir)
        base_name = Path(trial_dir).name
    else:
        base_name = "base"
    config["vizdom_name"] = config["ray_experiment_name"] + "_" + base_name

    # wrap config in ConfigDict
    config = ConfigDict(config)

    print(config)
    if config.policy == "gaussian":
        trainer_cls = BaseRLTrainer
    elif config.policy == "vae":
        trainer_cls = VAERLTrainer
    else:
        raise ValueError(f"Policy {config.policy} not implemented")

    trainer = trainer_cls(config)
    if config.mode == "train":
        trainer.train()
    elif config.mode == "eval":
        trainer.eval()


def main(_):
    config = _CONFIG.value.to_dict()
    if config["smoke_test"] is False:
        # run with ray tune
        param_space = {
            # "latent_dim": tune.grid_search([5, 8]),
            # "kl_div_weight": tune.grid_search([1.0, 1e-1, 1e-2]),
            "entropy_weight": tune.grid_search([0.0]),
            "seed": tune.grid_search([0, 1, 2]),
            "policy_lr": tune.grid_search([3e-4, 1e-4]),
            "gamma": tune.grid_search([0.99, 0.999]),
        }
        config.update(param_space)
        train_model = tune.with_resources(train_model_fn, {"cpu": 4, "gpu": 0.1})

        run_config = RunConfig(
            name=config["ray_experiment_name"],
            local_dir="/scr/aliang80/online_rl_diffusion/ray_results",
            storage_path="/scr/aliang80/online_rl_diffusion/ray_results",
            log_to_file=True,
        )
        tuner = tune.Tuner(train_model, param_space=config, run_config=run_config)
        results = tuner.fit()
        print(results)
    else:
        # run without ray tune
        train_model_fn(config)


if __name__ == "__main__":
    app.run(main)
