"""
Train policy with RL algorithm

Usage:
python3 trainer.py \
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
from functools import partial
from flax.training.train_state import TrainState
from typing import Any
from tensorflow_probability.substrates import jax as tfp
import pickle
from pathlib import Path
import utils
from models import policy_fn, actor_critic_fn
from ray import train, tune
from ray.train import RunConfig, ScalingConfig
import cv2
from trainer import BaseRLTrainer
from flax import traverse_util

eps = jnp.finfo(jnp.float32).eps
dist = tfp.distributions


class VAERLTrainer(BaseRLTrainer):
    def __init__(self, config: FrozenConfigDict):
        super().__init__(config)

    def policy_loss_fn(self, params, ts, rng_key, trajectory, returns):
        # compute reinforce loss
        action_output, value_estimate = ts.apply_fn(params, rng_key, trajectory.states)
        action_dist = action_output.action_dist
        latent_dist = action_output.latent_dist

        # compute action log probabilities
        # log p(a|z,s)
        logp_a_z_s = action_dist.log_prob(trajectory.actions).squeeze()

        # kl loss to uniform prior
        # KL(q(z|s) || p(z))
        prior = dist.Normal(0, 1)
        posterior = latent_dist
        kl_div = self.config.kl_div_weight * dist.kl_divergence(posterior, prior)
        # sum over latent dimensions
        kl_div = kl_div.sum(axis=-1)

        # log p(a|s), this is the ELBO
        logp_a_s = logp_a_z_s - kl_div

        # apply whitening to returns
        if self.config.baseline:
            advantage = returns - value_estimate
        else:
            returns = (returns - jnp.mean(returns)) / (jnp.std(returns) + eps)
            # advantage = jax.lax.stop_gradient(returns - value_estimate)
            advantage = returns

        pg_loss = -logp_a_s * advantage
        pg_loss = pg_loss.sum()

        # add exploration bonus, want to maximize entropy
        entropy_loss = self.config.entropy_weight * action_dist.entropy().sum()

        total_loss = pg_loss - entropy_loss

        metrics = {
            "entropy_loss": entropy_loss,
            "kl_div": kl_div.mean(),
            "elbo": logp_a_s.mean(),
        }

        if self.config.baseline:
            # compute value loss
            value_loss = jnp.mean((value_estimate - returns) ** 2)
            total_loss += self.config.value_coeff * value_loss
            metrics["value_loss"] = value_loss

        return total_loss, metrics
