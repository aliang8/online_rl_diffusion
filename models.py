import jax
import chex
import dataclasses
import haiku as hk
import jax.numpy as jnp
from typing import NamedTuple
from flax import linen as nn
from tensorflow_probability.substrates import jax as tfp

dist = tfp.distributions


@chex.dataclass
class ActionOutput:
    action: jnp.ndarray
    action_dist: dist.Normal


@chex.dataclass
class VAEActionOutput:
    action: jnp.ndarray
    action_dist: dist.Normal
    latent_dist: dist.Normal


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


@dataclasses.dataclass
class Policy(hk.Module):
    """
    Simple policy.

    2 hidden layers with leaky relu activations.
    Output layer predicts mean and log_stddev of normal distribution.
    """

    hidden_size: int
    action_dim: int
    is_continuous: bool

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

        if self.is_continuous:
            # predict mean and logvar
            mean = hk.Linear(self.action_dim)(x)
            logvar = hk.Linear(self.action_dim)(x)
            # clamp logvar
            logvar = jnp.clip(logvar, -20, 2)
            std = jnp.exp(logvar) ** 0.5
            action_dist = dist.Normal(loc=mean, scale=std)
        else:
            logits = hk.Linear(self.action_dim)(x)
            action_dist = dist.Categorical(logits=logits)

        # sample action from normal distribution
        action = action_dist.sample(seed=hk.next_rng_key())

        # apply tanh to action
        action = jnp.tanh(action)
        return ActionOutput(action=action, action_dist=action_dist)


class ActorCritic(hk.Module):
    def __init__(
        self,
        action_dim: int,
        hidden_size: int,
        is_continuous: bool,
        policy_cls: str,
        latent_dim: int = 8,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.hidden_size = hidden_size

        if policy_cls == "gaussian":
            self.actor = Policy(hidden_size, action_dim, is_continuous)
        else:
            self.actor = VAEPolicy(hidden_size, action_dim, latent_dim, is_continuous)

        self.critic = ValueFunction(hidden_size)

    def __call__(self, x: jnp.ndarray) -> NamedTuple:
        actor_output = self.actor(x)
        value = self.critic(x)
        return actor_output, value


class SharedActorCritic(hk.Module):
    def __init__(
        self,
        action_dim: int,
        hidden_size: int,
        is_continuous: bool,
        policy_cls: str,
        latent_dim: int = 8,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.hidden_size = hidden_size
        self.is_continuous = is_continuous

        self.backbone = hk.Sequential(
            [
                hk.Linear(hidden_size),
                jax.nn.leaky_relu,
                hk.Linear(hidden_size),
                jax.nn.leaky_relu,
            ]
        )
        if is_continuous:
            self.actor_mean = hk.Linear(action_dim)
            self.actor_logvar = hk.Linear(action_dim)
        else:
            self.actor_logits = hk.Linear(action_dim)

        self.critic = hk.Linear(1)

    def __call__(self, x: jnp.ndarray) -> NamedTuple:
        x = self.backbone(x)

        if self.is_continuous:
            mean = self.actor_mean(x)
            logvar = self.actor_logvar(x)
            logvar = jnp.clip(logvar, -20, 2)
            std = jnp.exp(logvar) ** 0.5
            action_dist = dist.Normal(loc=mean, scale=std)
            action = action_dist.sample(seed=hk.next_rng_key())
            action = jnp.tanh(action)
        else:
            logits = self.actor_logits(x)
            action_dist = dist.Categorical(logits=logits)
            action = action_dist.sample(seed=hk.next_rng_key())

        value = self.critic(x)

        if self.policy_cls == "vae":
            action_output = VAEActionOutput(action, action_dist, latent_dist)
        else:
            action_output = ActionOutput(action, action_dist)
        return action_output, value


class VAEPolicy(hk.Module):
    """
    Simple VAE policy.

    Approximate log p(a|s) with variational lower bound.
    Encoder q(z|s) and decoder p(a|z,s).

    log p(a|s) \geq ELBO = E_{q(z|s)}[log p(a|z,s)] - KL[q(z|s) || p(z)]

    We only use decoder during inference time.
    """

    def __init__(
        self,
        hidden_size: int,
        action_dim: int,
        latent_dim: int,
        is_continuous: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.is_continuous = is_continuous

        # encoder q(z|s)
        self.encoder = hk.Sequential(
            [
                hk.Linear(self.hidden_size),
                jax.nn.leaky_relu,
                hk.Linear(self.hidden_size),
                jax.nn.leaky_relu,
            ]
        )

        # decoder p(a|z,s)
        self.decoder = hk.Sequential(
            [
                hk.Linear(self.hidden_size),
                jax.nn.leaky_relu,
                hk.Linear(self.hidden_size),
                jax.nn.leaky_relu,
            ]
        )

    def __call__(self, state: jnp.ndarray) -> jnp.ndarray:
        state_embed = self.encoder(state)

        # predict mean and log_stddev
        mean = hk.Linear(self.latent_dim)(state_embed)
        logvar = hk.Linear(self.latent_dim)(state_embed)
        # clamp logvar
        logvar = jnp.clip(logvar, -20, 2)
        std = jnp.exp(logvar) ** 0.5
        latent_dist = dist.Normal(loc=mean, scale=std)

        # sample latent from latent distribution
        latent = latent_dist.sample(seed=hk.next_rng_key())

        # predict action from latent and state
        action = self.decoder(jnp.concatenate([latent, state], axis=-1))

        if self.is_continuous:
            action_mean = hk.Linear(self.action_dim)(action)
            action_logvar = hk.Linear(self.action_dim)(action)
            action_logvar = jnp.clip(action_logvar, -20, 2)
            action_std = jnp.exp(action_logvar) ** 0.5
            action_dist = dist.Normal(loc=action_mean, scale=action_std)
        else:
            logits = hk.Linear(self.action_dim)(action)
            action_dist = dist.Categorical(logits=logits)

        action = action_dist.sample(seed=hk.next_rng_key())
        return VAEActionOutput(
            action=action, action_dist=action_dist, latent_dist=latent_dist
        )


@hk.transform
def policy_fn(obs, hidden_size, action_dim):
    return Policy(hidden_size, action_dim)(obs)


@hk.transform
def vae_policy_fn(obs, hidden_size, action_dim, latent_dim):
    return VAEPolicy(hidden_size, action_dim, latent_dim)(obs)


@hk.transform
def value_fn(obs, hidden_size):
    return ValueFunction(hidden_size)(obs)


@hk.transform
def actor_critic_fn(
    obs,
    hidden_size,
    action_dim,
    share_backbone=False,
    is_continuous=True,
    policy_cls="gaussian",
    latent_dim=8,
):
    if share_backbone:
        return SharedActorCritic(
            action_dim, hidden_size, is_continuous, policy_cls, latent_dim
        )(obs)
    else:
        return ActorCritic(
            action_dim, hidden_size, is_continuous, policy_cls, latent_dim
        )(obs)
