"""
Train policy with RL algorithm

Usage:
python3 trainer.py \
    --config=configs/rl_config.py \
    --config.mode=train \
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
from ml_collections import ConfigDict, FieldReference, FrozenConfigDict, config_flags
from functools import partial
from flax.training.train_state import TrainState
from typing import Any
from tensorflow_probability.substrates import jax as tfp
import pickle
from pathlib import Path
import diffusion_rl.utils.utils as utils
from diffusion_rl.models.models import policy_fn, actor_critic_fn
from ray import train, tune
from ray.train import RunConfig, ScalingConfig
import cv2
from diffusion_rl.trainers.trainer import BaseRLTrainer
from flax import traverse_util

eps = jnp.finfo(jnp.float32).eps
dist = tfp.distributions


class VAERLTrainer(BaseRLTrainer):
    def __init__(self, config: FrozenConfigDict):
        super().__init__(config)

    def policy_loss_fn(self, params, ts, rng_key, trajectory):
        logging.info("inside policy loss")

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
        returns = trajectory.returns
        if self.config.baseline:
            advantage = returns - value_estimate
        else:
            returns = trajectory.returns
            # take mean of masked returns
            returns_mean = jnp.mean(returns, where=trajectory.mask)
            returns_std = jnp.std(returns, where=trajectory.mask)
            # apply whitening to returns
            returns = (returns - returns_mean) / (returns_std + eps)
            advantage = returns

        pg_loss = -logp_a_s * advantage
        pg_loss *= trajectory.mask
        pg_loss = pg_loss.sum()

        # add exploration bonus, want to maximize entropy
        entropy = action_dist.entropy()
        entropy *= trajectory.mask
        entropy_loss = self.config.entropy_weight * entropy.sum()

        total_loss = pg_loss - entropy_loss

        # compute prior loss
        # KL(p(z|s), q(z|s))
        prior_dist = action_output.prior_dist
        prior_kl_div = dist.kl_divergence(prior_dist, latent_dist)
        prior_kl_div = prior_kl_div.sum(axis=-1)
        prior_kl_div *= trajectory.mask
        prior_kl_div = prior_kl_div.sum()
        total_loss += self.config.prior_kl_weight * prior_kl_div

        metrics = {
            "entropy": jnp.mean(entropy, where=trajectory.mask),
            "entropy_loss": entropy_loss,
            "kl_div": kl_div.mean(),
            "elbo": logp_a_s.mean(),
            "prior_loss": prior_kl_div,
            "action_log_prob": logp_a_z_s.mean(),
        }

        if self.config.baseline:
            # compute value loss
            value_loss = (value_estimate - returns) ** 2
            value_loss *= trajectory.mask
            value_loss = value_loss.mean()
            total_loss += self.config.value_coeff * value_loss
            metrics["value_loss"] = value_loss

        return total_loss, metrics
