from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
from jax import random
import numpy as np
import optax
from typing import Any, Generator, Union
from flax.core import FrozenDict
from flax.training import checkpoints, common_utils, train_state
from absl import logging
from ml_collections import ConfigDict, FieldReference
import optax
import wandb
from pathlib import Path
import flax.linen as nn


# schedulers
def create_lr_schedule(schedule_type: str, **kwargs):
    if schedule_type == "constant":
        return optax.constant_schedule(**kwargs)
    elif schedule_type == "constant_warmup":
        return _constant_with_warmup(**kwargs)
    elif schedule_type == "cosine":
        return _cosine_with_warmup(**kwargs)
    else:
        raise NotImplementedError(schedule_type)


def _constant_with_warmup(value: float, warmup_steps: int):
    warmup = optax.linear_schedule(0, value, warmup_steps)
    constant = optax.constant_schedule(value=value)
    return optax.join_schedules([warmup, constant], boundaries=[warmup_steps])


def _cosine_with_warmup(
    init_value: float,
    peak_value: float,
    warmup_steps: int,
    decay_steps: int,
    decay_factor: float,
):
    return optax.warmup_cosine_decay_schedule(
        init_value=init_value,
        peak_value=peak_value,
        warmup_steps=warmup_steps,
        decay_steps=decay_steps,
        end_value=peak_value / decay_factor,
    )


# optimizer
def create_optimizer(
    optimizer_type: str,
    lr_schedule: optax.Schedule,
    max_grad_norm: float,
    grac_acc_steps: int,
    **kwargs,
):
    optimizer = None
    if optimizer_type == "adam":
        optimizer = optax.adam(learning_rate=lr_schedule, **kwargs)
    else:
        raise NotImplementedError(optimizer_type)
    optimizer = optax.chain(
        optimizer,
        optax.clip_by_global_norm(max_grad_norm),
    )
    optimizer = optax.MultiSteps(optimizer, grac_acc_steps)
    return optimizer


# types
ndarray = Union[jnp.ndarray, np.ndarray]
Dtype = Any
Rng = jax.random.PRNGKey
Params = optax.Params
Config = Any
ReplicatedState = Any


class Batch:
    image: ndarray
    label: ndarray | None


Dataset = Generator[Batch, None, None]


def get_config() -> Config:
    config = ConfigDict()

    config.project_name = "online_rl_diffusion"
    config.restore = ""
    config.dry_run = True
    config.log_level = logging.INFO

    seed = 42
    d_model = FieldReference(8)
    grad_acc = FieldReference(1)
    steps = FieldReference(30000)

    config.seed = seed
    config.effective_steps = steps
    config.steps = steps * grad_acc
    config.ckpt_dir = "results"
    config.log_interval = 100
    config.ckpt_interval = 5000
    config.eval_interval = 1000

    config.experiment_kwargs = ConfigDict(
        dict(
            config=dict(
                dry_run=config.dry_run,
                seed=seed,
                half_precision=False,
                diffusion=dict(
                    T=100,
                    beta_1=1e-4,
                    beta_T=0.02,
                ),
                train=dict(
                    ema_step_size=1 - 0.9995,
                    optimizer=dict(
                        optimizer_type="adam",
                        kwargs=dict(
                            max_grad_norm=1.0,
                            grac_acc_steps=grad_acc,
                        ),
                    ),
                    lr_schedule=dict(
                        schedule_type="constant",
                        kwargs=dict(value=1e-4),
                    ),
                ),
                model=dict(
                    model_kwargs=dict(
                        dim_init=d_model,
                        sinusoidal_embed_dim=d_model,
                        time_embed_dim=4 * d_model,
                    ),
                ),
            ),
        )
    )

    config.lock()

    return config


class TrainState(train_state.TrainState):
    ema_params: FrozenDict[str, Any]
    ema_step_size: float

    def apply_gradients(self, *, grads, **kwargs):
        next_state = super().apply_gradients(grads=grads, **kwargs)
        new_ema_params = optax.incremental_update(
            new_tensors=next_state.params,
            old_tensors=self.ema_params,
            step_size=self.ema_step_size,
        )
        return next_state.replace(ema_params=new_ema_params)


class SinusoidalPosEmbedding(nn.Module):
    dim: int

    @nn.compact
    def __call__(self, pos):
        """Refer to https://arxiv.org/pdf/1706.03762.pdf#subsection.3.5"""
        batch_size = pos.shape[0]

        assert self.dim % 2 == 0, self.dim
        assert pos.shape == (batch_size, 1), pos.shape

        d_model = self.dim // 2
        i = jnp.arange(d_model)[None, :]

        pos_embedding = pos * jnp.exp(-(2 * i / d_model) * jnp.log(10000))
        pos_embedding = jnp.concatenate(
            (jnp.sin(pos_embedding), jnp.cos(pos_embedding)), axis=-1
        )

        assert pos_embedding.shape == (batch_size, self.dim), pos_embedding.shape

        return pos_embedding


class TimeEmbedding(nn.Module):
    dim: int
    sinusoidal_embed_dim: int
    dtype: Dtype

    @nn.compact
    def __call__(self, time):
        x = SinusoidalPosEmbedding(self.sinusoidal_embed_dim)(time)
        x = nn.Dense(self.dim, dtype=self.dtype)(x)
        x = nn.gelu(x)
        x = nn.Dense(self.dim, dtype=self.dtype)(x)
        return x


class TimeConditionedMLP(nn.Module):
    dim: int
    dtype: Dtype

    @nn.compact
    def __call__(self, x: ndarray, t: ndarray, train: bool, rng: Rng | None = None):
        t = TimeEmbedding(self.dim, self.dim, self.dtype)(t)
        x = jnp.concatenate([x, t], axis=-1)
        x = nn.Sequential(
            [
                nn.Dense(self.dim, dtype=self.dtype),
                nn.gelu,
                nn.Dense(self.dim, dtype=self.dtype),
                nn.gelu,
            ]
        )(x)
        x = nn.Dense(1, dtype=self.dtype)(x)
        return x


class Diffuser:
    def __init__(self, eps_fn, diffuser_config: Config):
        self.eps_fn = eps_fn
        self.config = diffuser_config
        beta1 = diffuser_config.beta_1
        betaT = diffuser_config.beta_T
        T = diffuser_config.T
        self.betas = jnp.linspace(beta1, betaT, T, dtype=jnp.float32)
        self.alphas = 1 - self.betas
        self.alpha_bars = jnp.cumprod(self.alphas)

    @property
    def steps(self) -> int:
        return self.config.T

    def timesteps(self, steps: int):
        timesteps = jnp.linspace(0, self.steps, steps + 1)
        timesteps = jnp.rint(timesteps).astype(jnp.int32)
        return timesteps[::-1]

    @partial(jax.jit, static_argnums=(0,))
    def forward(self, x_0: ndarray, rng: Rng):
        """See algorithm 1 in https://arxiv.org/pdf/2006.11239.pdf"""
        rng1, rng2 = random.split(rng)
        t = random.randint(rng1, (len(x_0), 1), 0, self.steps)
        x_t, eps = self.sample_q(x_0, t, rng2)
        t = t.astype(x_t.dtype)
        return x_t, t, eps

    def sample_q(self, x_0: ndarray, t: ndarray, rng: Rng):
        """Samples x_t given x_0 by the q(x_t|x_0) formula."""
        alpha_t_bar = self.alpha_bars[t]
        eps = random.normal(rng, shape=x_0.shape, dtype=x_0.dtype)
        x_t = (alpha_t_bar**0.5) * x_0 + ((1 - alpha_t_bar) ** 0.5) * eps
        return x_t, eps

    @partial(jax.jit, static_argnums=(0,))
    def ddpm_reverse(self, params: Params, x_t: ndarray, t: int, rng: Rng):
        """See algorithm 2 in https://arxiv.org/pdf/2006.11239.pdf"""
        alpha_t = self.alphas[t]
        alpha_t_bar = self.alpha_bars[t]
        sigma_t = self.betas[t] ** 0.5

        z = (t > 0) * random.normal(rng, shape=x_t.shape, dtype=x_t.dtype)
        eps = self.eps_fn(params, x_t, t, train=False)

        x = (1 / alpha_t**0.5) * (
            x_t - ((1 - alpha_t) / (1 - alpha_t_bar) ** 0.5) * eps
        ) + sigma_t * z

        return x

    def ddpm_reverse(self, params: Params, x_T: ndarray, rng: Rng) -> ndarray:
        x = x_T

        for t in range(self.steps - 1, -1, -1):
            rng, rng_ = random.split(rng)
            x = self.ddpm_reverse(params, x, t, rng_)

        return x

    @partial(jax.jit, static_argnums=(0,))
    def ddim_reverse_step(
        self, params: Params, x_t: ndarray, t: ndarray, t_next: ndarray
    ):
        """See section 4.1 and C.1 in https://arxiv.org/pdf/2010.02502.pdf

        Note: alpha in the DDIM paper is actually alpha_bar in DDPM paper

        p)(x_t-1|x_t)

        I think here, we assume that sigma_t = 0 in the formula
        """
        alpha_t = self.alpha_bars[t]
        alpha_t_next = self.alpha_bars[t_next]

        eps = self.eps_fn(params, x_t, t, train=False)

        x_0 = (x_t - (1 - alpha_t) ** 0.5 * eps) / alpha_t**0.5
        x_t_direction = (1 - alpha_t_next) ** 0.5 * eps
        x_t_next = alpha_t_next**0.5 * x_0 + x_t_direction

        return x_t_next

    def ddim_reverse(self, params: Params, x_T: ndarray, steps: int):
        x = x_T

        # timesteps in reverse order
        ts = self.timesteps(steps)

        # t_next is actually the previous time step
        for t, t_next in zip(ts[:-1], ts[1:]):
            x = self.ddim_reverse_step(params, x, t, t_next)

        return x

    @staticmethod
    def expand_t(t: int, x: ndarray):
        return jnp.full((len(x), 1), t, dtype=x.dtype)


class Trainer:
    def __init__(self, rng: Rng, config: Config):
        self.global_step = 0
        self._init_rng, self._step_rng = random.split(rng)
        self._config = config
        self._diffuser = Diffuser(self._forward_fn, config.diffusion)
        self.action_dim = 1

        self._n_devices = jax.local_device_count()
        platform = jax.local_devices()[0].platform
        self._dtype = jnp.float32
        devices = jax.devices()

        logging.info(f"Devices: {devices}")
        logging.info(f"Device count: {self._n_devices}")
        logging.info(f"Running on platform: {platform}")
        logging.info(f"Using data type: {self._dtype}")

        # Initialized at first
        self._state, self._lr_schedule = self._create_state()
        self.device = jax.devices()[0]
        self._state = jax.device_put(self._state, self.device)

        if not config.dry_run:
            wandb.login()
            self.wandb_run = wandb.init(
                mode="disabled" if config.dry_run else "online",
                project=config.project_name,
                dir=str(Path.cwd() / "_wandb"),
                config=config.experiment_kwargs.config.to_dict(),
            )
        else:
            self.wandb_run = None

    def step(self):
        """Performs one training step"""
        self._step_rng, rng = random.split(self._step_rng)
        self._state, metrics = self._update_fn(self._state, rng)

        self.global_step += 1

        meta = {
            "step": self.global_step,
            "learning_rate": self._lr_schedule(self.global_step),
        }
        return metrics, meta

    def sample(self, num: int, steps: int, rng: Rng):
        shape = (num, self.action_dim)
        # start from random noise
        x_T = random.normal(rng, shape, dtype=self._dtype)
        # reverse the diffusion chain to get sample from p(x)
        x_0 = self._diffuser.ddim_reverse(self._state.ema_params, x_T, steps)
        # since this is actions, clip to [-1, 1]
        return jnp.clip(x_0, -1, 1)

    def save_checkpoint(self, ckpt_dir: str):
        checkpoints.save_checkpoint(
            ckpt_dir=ckpt_dir,
            target=self._state,
            step=self._state.step,
            keep=1,
        )

    def _create_state(self):
        rng_diffusion, rng_param, rng_dropout, rng_init = random.split(
            self._init_rng, 4
        )

        # batch size 1 and action_dim 1, sample model input
        x_0 = jax.random.normal(rng_init, (1, 1), dtype=self._dtype)
        x_t, t, _ = self._diffuser.forward(x_0, rng_diffusion)
        model = TimeConditionedMLP(dim=16, dtype=self._dtype)

        init_rngs = {"params": rng_param, "dropout": rng_dropout}
        params = model.init(init_rngs, x_t, t, train=True)
        count = sum(x.size for x in jax.tree_util.tree_leaves(params))
        logging.info(f"Parameter count: {count}")

        tx, lr_schedule = self._create_optimizer()

        state = TrainState.create(
            apply_fn=model.apply,
            params=params,
            ema_params=params,
            ema_step_size=self._config.train.ema_step_size,
            tx=tx,
        )
        return state, lr_schedule

    def _forward_fn(
        self,
        params: Params,
        x: ndarray,
        t: ndarray,
        train: bool,
        rng: Rng | None = None,
    ):
        rngs = {"dropout": rng} if rng is not None else None

        if t.size == 1:
            t = jnp.full((len(x), 1), t, dtype=x.dtype)

        assert self._state is not None
        return self._state.apply_fn(params, x, t, train, rngs=rngs)

    def reward(self, x: ndarray):
        # bandit reward function, a gaussian centered at 0
        y1 = jnp.clip(-18 * (0.5 * x) ** 2 + 1.1, a_min=0)
        return y1

    def _loss_fn(self, params: Params, x: ndarray, t: ndarray, eps: ndarray, rng: Rng):
        pred = self._forward_fn(params, x, t, train=True, rng=rng)

        # bandit reward, with reward_weight
        r = self.reward(x) * 0.1

        # loss is the l2 between predicted noise and actual noise
        ddpm_loss = optax.l2_loss(pred, eps)

        # policy gradient loss is -(ddpm_loss * reward)
        loss = -(ddpm_loss * r).mean()

        metrics = {
            "ddpm_loss": ddpm_loss.mean(),
            "reward": r.mean(),
            "policy_loss": loss,
        }
        return loss, metrics

    @partial(jax.jit, static_argnums=0)
    def _update_fn(self, state: TrainState, rng: Rng):
        rng1, rng2, rng3 = random.split(rng, 3)
        grad_loss_fn = jax.grad(self._loss_fn, has_aux=True)

        # sample new actions
        x_0 = self.sample(256, self._config.diffusion.T, rng1)

        # compute forward process
        x_t, t, eps = self._diffuser.forward(x_0, rng2)

        # compute noise reconstruction loss objective
        grads, metrics = grad_loss_fn(state.params, x_t, t, eps, rng3)
        state = state.apply_gradients(grads=grads)
        return state, metrics

    def _create_optimizer(self):
        op_config = self._config.train.optimizer
        lr_config = self._config.train.lr_schedule

        lr_schedule = create_lr_schedule(
            lr_config.schedule_type,
            **lr_config.kwargs,
        )
        optimizer = create_optimizer(
            optimizer_type=op_config.optimizer_type,
            lr_schedule=lr_schedule,
            **op_config.kwargs,
        )

        return optimizer, lr_schedule
