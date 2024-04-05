import numpy as np
import jax
import haiku as hk
import jax.numpy as jnp
from ml_collections import config_dict
from diffusion_rl.models.diffusion.model import DiffusionDDPM


@hk.transform
def diffusion_apply_fn(
    config: config_dict.ConfigDict,
    states: jnp.ndarray,
    output_dim: int,
):
    model = DiffusionDDPM(config, output_dim)
    return model(cond=states)


@hk.transform
def diffusion_compute_loss_fn(
    config: config_dict.ConfigDict,
    states: jnp.ndarray,
    actions: jnp.ndarray,
    output_dim: int,
):
    model = DiffusionDDPM(config, output_dim)
    return model.loss(actions, states)


def init_params(
    config: config_dict.ConfigDict,
    rng: jax.random.PRNGKey,
    state_dim: int,
    output_dim: int,
):
    dummy_states = jnp.zeros((1, state_dim))
    params = diffusion_apply_fn.init(
        rng, config=config, states=dummy_states, output_dim=output_dim
    )
    return params
