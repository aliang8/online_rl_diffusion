import numpy as np
import jax
import haiku as hk
import jax.numpy as jnp
from ml_collections import config_dict
from diffusion_rl.models.diffusion.model import Diffusion


@hk.transform
def diffusion_apply_fn(
    config: config_dict.ConfigDict,
    states: jnp.ndarray,
    state_dim: int,
    action_dim: int,
    max_action: int,
):
    model = Diffusion(config, state_dim, action_dim, max_action)
    return model(states)


def init_params(
    config: config_dict.ConfigDict,
    rng: jax.random.PRNGKey,
    state_dim: int,
    action_dim: int,
    max_action: int,
):
    dummy_states = jnp.zeros((1, state_dim))
    params = diffusion_apply_fn.init(
        rng, config, dummy_states, state_dim, action_dim, max_action
    )
    return params
