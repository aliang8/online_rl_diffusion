import jax
import dataclasses
import haiku as hk
import jax.numpy as jnp
from typing import NamedTuple


@dataclasses.dataclass
class Encoder(hk.Module):
    """Encoder model."""

    latent_size: int
    hidden_size1: int
    hidden_size2: int

    def __call__(self, x: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Encodes an image as an isotropic Guassian latent code."""
        x = hk.Flatten()(x)

        x = hk.Sequential(
            [
                hk.Linear(self.hidden_size1),
                jax.nn.relu,
                hk.Linear(self.hidden_size2),
                jax.nn.relu,
            ]
        )(x)
        mean = hk.Linear(self.latent_size)(x)
        log_stddev = hk.Linear(self.latent_size)(x)
        stddev = jnp.exp(log_stddev)

        return mean, stddev


@dataclasses.dataclass
class Decoder(hk.Module):
    """Decoder model."""

    hidden_size1: int
    hidden_size2: int
    action_dim: int

    def __call__(self, z: jax.Array) -> jax.Array:
        """Decodes a latent code into action samples."""
        output = hk.Sequential(
            [
                hk.Linear(self.hidden_size1),
                jax.nn.relu,
                hk.Linear(self.hidden_size2),
                jax.nn.relu,
                hk.Linear(self.action_dim),
                jax.nn.tanh,
            ]
        )(z)
        return output


class VAEOutput(NamedTuple):
    input: jax.Array
    mean: jax.Array
    variance: jax.Array
    recon: jax.Array


@dataclasses.dataclass
class VAE(hk.Module):
    """Main VAE model class, uses Encoder & Decoder under the hood."""

    encoder: Encoder
    decoder: Decoder

    def __call__(self, x: jnp.ndarray) -> VAEOutput:
        x = x.astype(jnp.float32)
        mean, stddev = self.encoder(x)

        # reparameterization trick
        z = mean + stddev * jax.random.normal(hk.next_rng_key(), mean.shape)

        # reconstruction
        recon = self.decoder(z)
        return VAEOutput(x, mean, jnp.square(stddev), recon)

    def sample(self, batch_size: int) -> jnp.ndarray:
        """Sample from the prior distribution."""
        z = jax.random.normal(hk.next_rng_key(), (batch_size, self.encoder.latent_size))
        return self.decoder(z)
