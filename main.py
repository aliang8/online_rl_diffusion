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
from ml_collections import ConfigDict, FieldReference, FrozenConfigDict, config_flags
from typing import Any
import pickle
from pathlib import Path
from ray import train, tune
from ray.train import RunConfig, ScalingConfig
from trainer import BaseRLTrainer
from vae_trainer import VAERLTrainer

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.01"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

_CONFIG = config_flags.DEFINE_config_file("config")

# shorthands for config parameters
psh = {
    "entropy_weight": "ew",
    "seed": "s",
    "policy_lr": "plr",
    "gamma": "g",
}


def train_model_fn(config):
    trial_dir = train.get_context().get_trial_dir()

    if trial_dir:
        # this is if we are running with Ray
        logging.info("trial dir: ", trial_dir)
        config["root_dir"] = Path(trial_dir)
        base_name = Path(trial_dir).name
        config["exp_name"] = base_name
        # the group name is without seed
        config["group_name"] = re.sub("_s-\d", "", base_name)
        logging.info(f"wandb group name: {config['group_name']}")
    else:
        suffix = f"{config['exp_name']}_s-{config['seed']}_t-{config['policy']}"
        config["root_dir"] = Path(config["root_dir"]) / "results" / suffix

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


param_space = {
    # "latent_dim": tune.grid_search([5, 8]),
    # "kl_div_weight": tune.grid_search([1.0, 1e-1, 1e-2]),
    "entropy_weight": tune.grid_search([0.0]),
    "seed": tune.grid_search([0, 1, 2]),
    "policy_lr": tune.grid_search([3e-4, 1e-4]),
    "gamma": tune.grid_search([0.99, 0.999]),
}


def trial_str_creator(trial):
    trial_str = trial.config["exp_name"] + "_"
    for k, v in trial.config.items():
        if k in psh and k in param_space:
            trial_str += f"{psh[k]}-{v}_"
    # trial_str += str(trial.trial_id)

    trial_str = trial_str[:-1]
    logging.info(f"trial_str: {trial_str}")
    return trial_str


def main(_):
    config = _CONFIG.value.to_dict()
    if config["smoke_test"] is False:
        # run with ray tune
        config.update(param_space)
        train_model = tune.with_resources(train_model_fn, {"cpu": 4, "gpu": 0.1})

        run_config = RunConfig(
            name=config["exp_name"],
            local_dir="/scr/aliang80/online_rl_diffusion/ray_results",
            storage_path="/scr/aliang80/online_rl_diffusion/ray_results",
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
