import jax
import dataclasses
import haiku as hk
import jax.numpy as jnp
from typing import NamedTuple
from flax import linen as nn
from tensorflow_probability.substrates import jax as tfp

dist = tfp.distributions


@dataclasses.dataclass
class Policy(hk.Module):
    hidden_size: int
    action_dim: int
    is_discrete: bool

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        net = hk.Sequential(
            [
                hk.Linear(self.hidden_size),
                jax.nn.leaky_relu,
                hk.Linear(self.hidden_size),
                jax.nn.leaky_relu,
                hk.Linear(self.action_dim),
            ]
        )
        logits = net(x)
        action = jax.random.categorical(hk.next_rng_key(), logits=logits)
        log_prob = dist.Categorical(logits=logits).log_prob(action)
        return action, logits, log_prob


@dataclasses.dataclass
class ValueFunction(hk.Module):
    """Value function model."""

    hidden_size: int

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Value function."""
        net = hk.Sequential(
            [
                hk.Linear(self.hidden_size),
                jax.nn.leaky_relu,
                hk.Linear(self.hidden_size),
                jax.nn.leaky_relu,
                hk.Linear(1),
            ]
        )
        return net(x)


@dataclasses.dataclass
class Encoder(hk.Module):
    """Encoder model."""

    latent_size: int
    hidden_size: int

    def log_prob(self, x=None, mu_e=None, log_var_e=None, z=None):
        if x is not None:
            mu_e, log_var_e = self.encode(x)
            z = self.sample(mu_e=mu_e, log_var_e=log_var_e)
        else:
            if (mu_e is None) or (log_var_e is None) or (z is None):
                raise ValueError("mu, log-scale and z can`t be None!")

        return log_normal_diag(z, mu_e, log_var_e)

    def __call__(
        self, obs: jax.Array, action: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        """
        Encodes state + action as an isotropic Guassian latent code.
        2-layer MLP with leaky ReLU activations.

        Outputs mean and std parameters of the Gaussian distribution.
        """

        # concatenate state and action
        x = jnp.concatenate([obs, action], axis=-1)
        x = hk.Sequential(
            [
                hk.Linear(self.hidden_size),
                jax.nn.leaky_relu,
                hk.Linear(self.hidden_size),
                jax.nn.leaky_relu,
            ]
        )(x)
        mean = hk.Linear(self.latent_size)(x)
        log_stddev = hk.Linear(self.latent_size)(x)
        stddev = jnp.exp(log_stddev)

        return mean, stddev


@dataclasses.dataclass
class Decoder(hk.Module):
    """Decoder model."""

    hidden_size: int
    action_dim: int
    is_discrete: bool

    def __call__(self, z: jax.Array, obs: jax.Array) -> jax.Array:
        """Decodes a latent code into action samples."""
        z = jnp.concatenate([z, obs], axis=-1)
        output = hk.Sequential(
            [
                hk.Linear(self.hidden_size),
                jax.nn.leaky_relu,
                hk.Linear(self.hidden_size),
                jax.nn.leaky_relu,
                hk.Linear(self.action_dim),
            ]
        )(z)

        if not self.is_discrete:
            output = jax.nn.tanh(output)

        return output


class VAEOutput(NamedTuple):
    input: jax.Array
    mean: jax.Array
    variance: jax.Array
    recon: jax.Array
    z: jax.Array
    kl_to_learned_prior: jax.Array
    kl: jax.Array


@dataclasses.dataclass
class StateConditionedPrior(hk.Module):
    """State-conditioned prior distribution."""

    latent_size: int
    hidden_size: int

    def __call__(self, obs: jax.Array) -> jax.Array:
        """Sample from the prior distribution."""
        net = hk.Sequential(
            [
                hk.Linear(self.hidden_size),
                jax.nn.leaky_relu,
                hk.Linear(self.hidden_size),
                jax.nn.leaky_relu,
            ]
        )
        hidden = net(obs)
        mean = hk.Linear(self.latent_size)(hidden)
        log_stddev = hk.Linear(self.latent_size)(hidden)
        stddev = jnp.exp(log_stddev)

        return mean, stddev

    def sample(self, obs: jax.Array) -> jax.Array:
        """Sample from the prior distribution."""
        mean, stddev = self(obs)
        z = mean + stddev * jax.random.normal(hk.next_rng_key(), mean.shape)
        return z


@dataclasses.dataclass
class VAE(hk.Module):
    """Main VAE model class, uses Encoder & Decoder under the hood."""

    encoder: Encoder
    decoder: Decoder
    prior: hk.Module

    def __call__(self, obs: jnp.ndarray, action: jnp.ndarray) -> VAEOutput:
        obs = obs.astype(jnp.float32)
        # action = action.astype(jnp.int32)

        # q(z| s, a)
        mean, stddev = self.encoder(obs, action)

        # reparameterization trick
        z = mean + stddev * jax.random.normal(hk.next_rng_key(), mean.shape)

        # reconstruction
        # predicts logits for discrete action
        # p(a | z, s)
        recon = self.decoder(z, obs)

        # compute kl to prior
        # p(z|s)
        prior_mu, prior_std = self.prior(obs)
        prior_dist = dist.Normal(loc=prior_mu, scale=prior_std)
        # stop gradient to posterior
        posterior_dist = dist.Normal(
            loc=jax.lax.stop_gradient(mean), scale=jax.lax.stop_gradient(stddev)
        )

        # KL(q(z|s, a) || p(z|s))
        # training the learned prior
        kl_to_learned_prior = dist.kl_divergence(posterior_dist, prior_dist).sum(axis=1)

        # compute KL to unit-variance Gaussian
        posterior_dist = dist.Normal(loc=mean, scale=stddev)
        kl_to_prior = dist.kl_divergence(posterior_dist, dist.Normal(0, 1)).sum(axis=1)

        return VAEOutput(
            action, mean, jnp.square(stddev), recon, z, kl_to_prior, kl_to_learned_prior
        )

        # x = x.astype(jnp.float32)
        # mean, stddev = self.encoder(x)

        # # reparameterization trick
        # z = mean + stddev * jax.random.normal(hk.next_rng_key(), mean.shape)

        # # reconstruction
        # recon = self.decoder(z)
        # prior_logp = (
        #     self.prior.log_prob(z)
        #     - self.encoder.log_prob(
        #         mu_e=mean, log_var_e=jnp.log(jnp.square(stddev)), z=z
        #     )
        # ).sum(axis=1)
        # return VAEOutput(action, mean, jnp.square(stddev), recon, z)

    # def sample(self, batch_size: int) -> jnp.ndarray:
    #     """Sample from the prior distribution."""
    #     z = self.prior.sample(batch_size)
    #     return self.decoder(z)

    def sample(self, obs: jnp.ndarray) -> jnp.ndarray:
        """Sample from the prior distribution."""
        z = self.prior.sample(obs)
        action = self.decoder(z, obs)
        return action


# flow prior
def log_standard_normal(x, reduction=None, axis=None):
    log_p = -0.5 * jnp.log(2.0 * jnp.pi) - 0.5 * x**2.0
    if reduction == "avg":
        return jnp.mean(log_p, axis=axis)
    elif reduction == "sum":
        return jnp.sum(log_p, axis=axis)
    else:
        return log_p


def log_normal_diag(x, mu, log_var, reduction=None, axis=None):
    log_p = (
        -0.5 * jnp.log(2.0 * jnp.pi)
        - 0.5 * log_var
        - 0.5 * jnp.exp(-log_var) * (x - mu) ** 2.0
    )
    if reduction == "avg":
        return jnp.mean(log_p, axis)
    elif reduction == "sum":
        return jnp.sum(log_p, axis)
    else:
        return log_p


class StandardPrior(nn.Module):
    def __init__(self, latent_size: int, action_dim: int):
        super().__init__()

        self.latent_size = latent_size
        self.action_dim = action_dim

        # params weights
        self.means = hk.get_parameter(
            "means", init=jnp.zeros, shape=(self.latent_size,)
        )
        self.logvars = hk.get_parameter(
            "logvars", init=jnp.ones, shape=(self.latent_size,)
        )

    def sample(self, batch_size):
        noise = jax.random.normal(hk.next_rng_key(), (batch_size, self.action_dim))
        return noise * jnp.exp(0.5 * self.logvars) + self.means

    def log_prob(self, z):
        # return log_standard_normal(z)
        return log_normal_diag(z, self.means, self.logvars)


class FlowPrior(hk.Module):
    """Flow-based prior distribution."""

    def __init__(
        self, latent_size: int, hidden_size: int, num_flows: int, action_dim: int
    ):
        super().__init__()
        self.latent_size = latent_size
        self.hidden_size = hidden_size
        self.num_flows = num_flows
        self.action_dim = action_dim

        # define flow
        net_s = lambda: hk.Sequential(
            [
                hk.Linear(self.hidden_size),
                jax.nn.leaky_relu,
                hk.Linear(self.hidden_size),
                jax.nn.leaky_relu,
                hk.Linear(self.latent_size // 2),
                jax.nn.tanh,
            ]
        )
        self.s = [net_s() for _ in range(self.num_flows)]

        net_t = lambda: hk.Sequential(
            [
                hk.Linear(self.hidden_size),
                jax.nn.leaky_relu,
                hk.Linear(self.hidden_size),
                jax.nn.leaky_relu,
                hk.Linear(self.latent_size // 2),
            ]
        )
        self.t = [net_t() for _ in range(self.num_flows)]

    def coupling(self, x, index, forward=True):
        """Coupling layer."""
        x0, x1 = jnp.split(x, 2, axis=1)
        s = self.s[index](x0)
        t = self.t[index](x0)
        if forward:
            y1 = (x1 - t) * jnp.exp(-s)
        else:
            y1 = x1 * jnp.exp(s) + t
        return jnp.concatenate([x0, y1], axis=1), s

    def f(self, x):
        log_det_J, z = jnp.zeros(x.shape[0]), x
        for i in range(self.num_flows):
            z, s = self.coupling(z, i, forward=True)
            z = z[:, ::-1]
            log_det_J -= s.sum(axis=-1)
        return z, log_det_J

    def f_inv(self, z):
        x = z
        for i in reversed(range(self.num_flows)):
            x = x[:, ::-1]
            x, _ = self.coupling(x, i, forward=False)
        return x

    def sample(self, batch_size: int) -> jnp.ndarray:
        """Sample from the prior distribution."""
        z = jax.random.normal(hk.next_rng_key(), (batch_size, self.action_dim))
        x = self.f_inv(z)
        return x.reshape(-1, self.action_dim)

    def log_prob(self, x):
        z, log_det_J = self.f(x)
        log_p = log_standard_normal(z) + log_det_J[..., None]
        return log_p
