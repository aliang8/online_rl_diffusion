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
import mlogger
from tensorflow_probability.substrates import jax as tfp
import pickle
from pathlib import Path
import utils
from models import policy_fn, vae_policy_fn
from ray import train, tune
from ray.train import RunConfig, ScalingConfig
import cv2
from trainer import BaseRLTrainer

eps = jnp.finfo(jnp.float32).eps
dist = tfp.distributions


class VAERLTrainer(BaseRLTrainer):
    def __init__(self, config: FrozenConfigDict):
        self.loss_keys = [
            ("entropy_loss", mlogger.metric.Average, "Entropy Loss"),
            ("kl_div", mlogger.metric.Average, "KL Loss"),
            ("elbo", mlogger.metric.Average, "ELBO"),
            ("return", mlogger.metric.Average, "Returns"),
            ("success", mlogger.metric.Average, "Success Rate"),
            ("episode_length", mlogger.metric.Average, "Episode Length"),
        ]
        super().__init__(config)

    def create_ts(self, rng):
        sample_obs = jnp.zeros((1, self.obs_dim))
        policy_params = vae_policy_fn.init(
            rng,
            sample_obs,
            hidden_size=self.config.hidden_size,
            latent_dim=self.config.latent_dim,
            action_dim=self.action_dim,
        )

        policy_opt = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adam(self.config.policy_lr),
        )

        param_count = sum(p.size for p in jax.tree_util.tree_leaves(policy_params))
        print(f"Number of policy parameters: {param_count}")

        policy_fn_apply = partial(
            jax.jit(vae_policy_fn.apply, static_argnums=(3, 4, 5)),
            hidden_size=self.config.hidden_size,
            latent_dim=self.config.latent_dim,
            action_dim=self.action_dim,
        )

        ts = TrainState.create(
            apply_fn=policy_fn_apply,
            params=policy_params,
            tx=policy_opt,
        )
        return ts

    def policy_loss_fn(self, params, ts, rng_key, trajectory, returns):
        # compute reinforce loss
        _, action_dist, _, latent_dist = ts.apply_fn(params, rng_key, trajectory.states)

        # compute action log probabilities
        # log p(a|z,s)
        logp_z_s = action_dist.log_prob(trajectory.actions).squeeze(axis=-1)

        # kl loss to uniform prior
        # KL(q(z|s) || p(z))
        prior = dist.Normal(0, 1)
        posterior = latent_dist
        kl_div = self.config.kl_div_weight * dist.kl_divergence(posterior, prior)
        # sum over latent dimensions
        kl_div = kl_div.sum(axis=-1)

        # log p(a|s), this is the ELBO
        logp_a_s = logp_z_s - kl_div

        # apply whitening to returns
        returns = (returns - jnp.mean(returns)) / (jnp.std(returns) + eps)
        # advantage = jax.lax.stop_gradient(returns - value_estimate)
        advantage = returns
        pg_loss = -(logp_a_s * advantage).sum()

        # add exploration bonus, want to maximize entropy
        entropy_loss = self.config.entropy_weight * action_dist.entropy().sum()

        total_loss = pg_loss - entropy_loss

        metrics = {
            "entropy_loss": entropy_loss,
            "kl_div": kl_div.mean(),
            "elbo": logp_a_s.mean(),
        }

        return total_loss, metrics
