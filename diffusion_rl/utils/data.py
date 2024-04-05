import chex
import optax
import jax
import d4rl
import dataclasses
import jax.numpy as jnp

# import gymnasium as gym
import gym
import numpy as np
from flax.training.train_state import TrainState
from typing import Any, Callable
from flax import core, struct
from changepoint_aug.density_estimation.data import load_pkl_dataset


@chex.dataclass
class Trajectory:
    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    next_states: np.ndarray
    success: bool
    returns: np.ndarray
    mask: np.ndarray


def make_env(env_name: str, seed: int = 0):
    env = gym.make(env_name, render_mode="rgb_array")

    print("observation space:", env.observation_space)
    print("action space:", env.action_space)

    # first reset to set seed
    env.reset(seed=seed)
    return env


@chex.dataclass
class Batch:
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    masks: np.ndarray
    next_observations: np.ndarray


class Dataset(object):
    def __init__(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        masks: np.ndarray,
        dones_float: np.ndarray,
        next_observations: np.ndarray,
        size: int,
    ):
        self.observations = observations
        self.actions = actions
        self.rewards = rewards
        self.masks = masks
        self.dones_float = dones_float
        self.next_observations = next_observations
        self.size = size

    def sample(self, batch_size: int) -> Batch:
        indx = np.random.randint(self.size, size=batch_size)
        return Batch(
            observations=self.observations[indx],
            actions=self.actions[indx],
            rewards=self.rewards[indx],
            masks=self.masks[indx],
            next_observations=self.next_observations[indx],
        )


class D4RLDataset(Dataset):
    def __init__(self, env: gym.Env, clip_to_eps: bool = True, eps: float = 1e-5):
        dataset = d4rl.qlearning_dataset(env)

        if clip_to_eps:
            lim = 1 - eps
            dataset["actions"] = np.clip(dataset["actions"], -lim, lim)

        dones_float = np.zeros_like(dataset["rewards"])

        for i in range(len(dones_float) - 1):
            if (
                np.linalg.norm(
                    dataset["observations"][i + 1] - dataset["next_observations"][i]
                )
                > 1e-6
                or dataset["terminals"][i] == 1.0
            ):
                dones_float[i] = 1
            else:
                dones_float[i] = 0

        dones_float[-1] = 1

        super().__init__(
            dataset["observations"].astype(np.float32),
            actions=dataset["actions"].astype(np.float32),
            rewards=dataset["rewards"].astype(np.float32),
            masks=1.0 - dataset["terminals"].astype(np.float32),
            dones_float=dones_float.astype(np.float32),
            next_observations=dataset["next_observations"].astype(np.float32),
            size=len(dataset["observations"]),
        )


def load_dataset(config, envs):
    if config.dataset == "d4rl":
        # load dataset
        dataset = D4RLDataset(envs)
    elif config.dataset == "maze":
        data_dir = (
            "/scr/aliang80/changepoint_aug/changepoint_aug/density_estimation/datasets"
        )
        data_file = "sac_maze_200.pkl"
        dataset, *_ = load_pkl_dataset(
            data_dir,
            data_file,
            config.num_trajs,
            config.batch_size,
            train_perc=1.0,
            env="MAZE",
            augmentation_data=[],
            num_augmentation_steps=0,
        )
        obs, next_obs, actions, next_actions, rewards, dones = dataset[:]
        dataset = Dataset(
            observations=np.array(obs),
            actions=np.array(actions),
            rewards=np.array(rewards),
            masks=1.0 - np.array(dones),
            dones_float=np.array(dones),
            next_observations=np.array(next_obs),
            size=obs.shape[0],
        )

    return dataset
