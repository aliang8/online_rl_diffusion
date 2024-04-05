import numpy as np
import jax
import haiku as hk
import jax.numpy as jnp
from ml_collections import config_dict
from diffusion_rl.models.diffusion.model import DiffusionDDPM


@hk.transform
def diffusion_apply_fn(
    config: config_dict.ConfigDict,
    output_dim: int,
    batch_size: int = None,
    cond: jnp.ndarray = None,
):
    model = DiffusionDDPM(config, output_dim)
    return model(batch_size=batch_size, cond=cond)


@hk.transform
def diffusion_compute_loss_fn(
    config: config_dict.ConfigDict,
    input: jnp.ndarray,
    cond: jnp.ndarray,
    output_dim: int,
):
    model = DiffusionDDPM(config, output_dim)
    return model.loss(x=input, cond=cond)


def init_params(
    config: config_dict.ConfigDict,
    rng: jax.random.PRNGKey,
    input_dim: int,
    output_dim: int,
):
    dummy_input = jnp.zeros((1, input_dim))
    params = diffusion_apply_fn.init(
        rng, config=config, cond=dummy_input, output_dim=output_dim
    )
    return params
