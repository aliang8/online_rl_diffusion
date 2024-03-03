import jax
import dataclasses
import haiku as hk
import jax.numpy as jnp
from typing import NamedTuple
from flax import linen as nn
from tensorflow_probability.substrates import jax as tfp

dist = tfp.distributions


@dataclasses.dataclass
class GaussianPolicy(hk.Module):
    """
    Simple Gaussian policy.

    2 hidden layers with leaky relu activations. Output layer predicts mean and log_stddev of normal distribution.
    """

    hidden_size: int
    action_dim: int

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        net = hk.Sequential(
            [
                hk.Linear(self.hidden_size),
                jax.nn.leaky_relu,
                hk.Linear(self.hidden_size),
                jax.nn.leaky_relu,
            ]
        )
        x = net(x)
        # predict mean and log_stddev
        mean = hk.Linear(self.action_dim)(x)
        log_stddev = hk.Linear(self.action_dim)(x)
        # clamp log_stddev
        log_stddev = jnp.clip(log_stddev, -20, 2)
        std = jnp.exp(log_stddev)
        action_dist = dist.Normal(loc=mean, scale=std)

        # sample action from normal distribution
        action = action_dist.sample(seed=hk.next_rng_key())

        # apply tanh to action
        action = jnp.tanh(action)
        log_prob = action_dist.log_prob(action)

        return action, action_dist, log_prob


@dataclasses.dataclass
class VAEPolicy(hk.Module):
    """
    Simple VAE policy.

    Approximate log p(a|s) with variational lower bound.
    Encoder q(z|s) and decoder p(a|z,s).

    log p(a|s) \geq ELBO = E_{q(z|s)}[log p(a|z,s)] - KL[q(z|s) || p(z)]

    We only use decoder during inference time.
    """

    hidden_size: int
    action_dim: int
    latent_dim: int

    def __call__(self, state: jnp.ndarray) -> jnp.ndarray:
        # encoder q(z|s)
        encoder = hk.Sequential(
            [
                hk.Linear(self.hidden_size),
                jax.nn.leaky_relu,
                hk.Linear(self.hidden_size),
                jax.nn.leaky_relu,
            ]
        )
        state_embed = encoder(state)
        # predict mean and log_stddev
        mean = hk.Linear(self.latent_dim)(state_embed)
        log_stddev = hk.Linear(self.latent_dim)(state_embed)
        # clamp log_stddev
        log_stddev = jnp.clip(log_stddev, -20, 2)
        std = jnp.exp(log_stddev)
        latent_dist = dist.Normal(loc=mean, scale=std)

        # sample latent from latent distribution
        latent = latent_dist.sample(seed=hk.next_rng_key())

        # decoder p(a|z,s)
        decoder = hk.Sequential(
            [
                hk.Linear(self.hidden_size),
                jax.nn.leaky_relu,
                hk.Linear(self.hidden_size),
                jax.nn.leaky_relu,
            ]
        )

        # predict action from latent and state
        action = decoder(jnp.concatenate([latent, state], axis=-1))
        action_mean = hk.Linear(self.action_dim)(action)
        action_log_stddev = hk.Linear(self.action_dim)(action)
        action_log_stddev = jnp.clip(action_log_stddev, -20, 2)
        action_std = jnp.exp(action_log_stddev)
        action_dist = dist.Normal(loc=action_mean, scale=action_std)
        action = action_dist.sample(seed=hk.next_rng_key())
        log_prob = action_dist.log_prob(action)

        return action, action_dist, log_prob, latent_dist


@dataclasses.dataclass
class ValueFunction(hk.Module):
    """
    Simple value function for computing baseline
    """

    hidden_size: int

    def __call__(self, state: jnp.ndarray) -> jnp.ndarray:
        net = hk.Sequential(
            [
                hk.Linear(self.hidden_size),
                jax.nn.leaky_relu,
                hk.Linear(self.hidden_size),
                jax.nn.leaky_relu,
                hk.Linear(1),
            ]
        )
        return net(state)


@hk.transform
def policy_fn(obs, hidden_size, action_dim):
    return GaussianPolicy(hidden_size, action_dim)(obs)


@hk.transform
def vae_policy_fn(obs, hidden_size, action_dim, latent_dim):
    return VAEPolicy(hidden_size, action_dim, latent_dim)(obs)


@hk.transform
def value_fn(obs, hidden_size):
    return ValueFunction(hidden_size)(obs)
