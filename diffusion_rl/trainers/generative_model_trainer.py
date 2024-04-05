import jax
import jax.numpy as jnp
import functools
import sklearn.datasets
import numpy as np
import chex
from ml_collections import ConfigDict, FrozenConfigDict
from diffusion_rl.trainers.offline_trainer import OfflineTrainer, create_ts
from diffusion_rl.models.diffusion.helpers import (
    diffusion_apply_fn,
    diffusion_compute_loss_fn,
)


class GenerativeModelTrainer(OfflineTrainer):
    def __init__(self, config: ConfigDict):
        super().__init__(config)

        # input to the diffusion model is state
        # dim = self.state_dim
        self.input_dim = 6

        self.ts_policy = create_ts(
            config,
            next(self.rng_seq),
            input_dim=self.input_dim,
            output_dim=self.input_dim,
        )

        self.compute_loss = functools.partial(
            jax.jit(
                diffusion_compute_loss_fn.apply,
                static_argnames=("config", "output_dim"),
            ),
            config=FrozenConfigDict(config.policy),
            output_dim=self.input_dim,
        )

        def loss_fn(params, ts, batch, rng):
            model_input = batch.observations[:, : self.input_dim]
            recon_loss = self.compute_loss(params, rng, input=model_input, cond=None)
            recon_loss = jnp.mean(recon_loss)

            metrics = {"recon_loss": recon_loss}
            return recon_loss, metrics

        def update_step(ts, batch, rng):
            (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(
                ts.params, ts, batch, rng
            )
            ts = ts.apply_gradients(grads=grads)
            return ts, metrics

        self.jit_update_step = jax.jit(update_step)
