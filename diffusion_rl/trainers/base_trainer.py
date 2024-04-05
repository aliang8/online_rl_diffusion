from absl import app, logging
import jax
import jax.numpy as jnp
import numpy as np
import haiku as hk
import os
import tqdm
import pickle
import time
import wandb
import optax
import einops
import d4rl
import jax.tree_util as jtu
from ml_collections import FrozenConfigDict
from pathlib import Path

# import gymnasium as gym
import gym  # d4rl uses the old gym interface
from jax import config as jax_config


class BaseTrainer:
    def __init__(self, config: FrozenConfigDict):
        self.config = config

        print(self.config)

        self.global_step = 0
        self.rng_seq = hk.PRNGSequence(config.seed)
        np.random.seed(config.seed)

        if self.config.use_wb:
            self.wandb_run = wandb.init(
                # set the wandb project where this run will be logged
                entity="glamor",
                project="diffusion_rl",
                name=self.config.exp_name,
                notes=self.config.notes,
                tags=self.config.tags,
                # track hyperparameters and run metadata
                config=self.config,
                group=self.config.group_name,
            )
        else:
            self.wandb_run = None

        # setup log dirs
        self.exp_dir = Path(self.config.exp_dir)
        print("experiment dir: ", self.exp_dir)

        self.ckpt_dir = self.exp_dir / "model_ckpts"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.video_dir = self.exp_dir / "videos"
        self.video_dir.mkdir(parents=True, exist_ok=True)

        # create env
        self.envs = gym.make(config.env_id).unwrapped
        self.envs.seed(config.seed)
        self.eval_envs = gym.make(config.env_id).unwrapped
        self.eval_envs.seed(config.seed + 100)

        self.obs_shape = self.envs.observation_space.shape
        self.state_dim = self.obs_shape[0]
        self.continuous_actions = not isinstance(
            self.envs.action_space, gym.spaces.Discrete
        )
        if isinstance(self.envs.action_space, gym.spaces.Discrete):
            self.action_dim = self.envs.action_space.n
            self.input_action_dim = 1
        else:
            self.input_action_dim = self.action_dim = self.envs.action_space.shape[0]

        if self.config.log_level == "info":
            logging.set_verbosity(logging.INFO)
        elif self.config.log_level == "debug":
            logging.set_verbosity(logging.DEBUG)

        if not self.config.enable_jit:
            jax_config.update("jax_disable_jit", True)

        logging.info(f"obs_shape: {self.obs_shape}, action_dim: {self.action_dim}")

    def create_ts(self):
        raise NotImplementedError

    def train_step(self, batch):
        raise NotImplementedError

    def test(self, epoch):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError
