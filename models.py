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
        log_prob = action_dist.log_prob(action)
        return action, action_dist, log_prob


@hk.transform
def policy_fn(obs, hidden_size, action_dim):
    return GaussianPolicy(hidden_size, action_dim)(obs)
