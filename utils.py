import chex
import optax
import jax
import dataclasses
import jax.numpy as jnp
import gymnasium as gym
import numpy as np
from flax.training.train_state import TrainState
from typing import Any, Callable
from flax import core, struct


def create_learning_rate_fn(
    num_epochs, warmup_epochs, base_learning_rate, steps_per_epoch
):
    """Creates learning rate schedule."""
    warmup_fn = optax.linear_schedule(
        init_value=0.0,
        end_value=base_learning_rate,
        transition_steps=warmup_epochs * steps_per_epoch,
    )
    cosine_epochs = max(num_epochs - warmup_epochs, 1)
    cosine_fn = optax.cosine_decay_schedule(
        init_value=base_learning_rate, decay_steps=cosine_epochs * steps_per_epoch
    )
    schedule_fn = optax.join_schedules(
        schedules=[warmup_fn, cosine_fn],
        boundaries=[warmup_epochs * steps_per_epoch],
    )
    return schedule_fn


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
