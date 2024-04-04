import copy
import numpy as np
import jax
import jax.numpy as jnp
import haiku as hk
from ml_collections import config_dict
from diffusion_rl.models.diffusion.utils import (
    linear_beta_schedule,
    cosine_beta_schedule,
    vp_beta_schedule,
    extract,
    SinusoidalPosEmb,
)


class NoiseModel(hk.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_size: int,
        time_embed_dim: int,
        w_init: hk.initializers.Initializer,
        b_init: hk.initializers.Initializer,
    ):
        super().__init__()
        init_kwargs = dict(w_init=w_init, b_init=b_init)
        self.time_mlp = hk.Sequential(
            [
                SinusoidalPosEmb(time_embed_dim),
                hk.Linear(time_embed_dim * 2),
                jax.nn.gelu,
                hk.Linear(time_embed_dim),
            ]
        )

        self.net = hk.Sequential(
            [
                hk.Linear(hidden_size, **init_kwargs),
                jax.nn.gelu,
                hk.Linear(hidden_size, **init_kwargs),
                jax.nn.gelu,
                hk.Linear(action_dim, **init_kwargs),
            ]
        )

    def __call__(self, x, time, state):
        t = self.time_mlp(time)
        x = jnp.concatenate([x, t, state], axis=1)
        x = self.net(x)
        return x


class Diffusion(hk.Module):
    def __init__(
        self,
        config: config_dict.ConfigDict,
        state_dim: int,
        action_dim: int,
        max_action: int,
        w_init: hk.initializers.Initializer = hk.initializers.VarianceScaling(2.0),
        b_init: hk.initializers.Initializer = hk.initializers.Constant(0.0),
    ):
        super().__init__()

        self.config = config
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action

        if self.config.beta_schedule == "linear":
            self.betas = linear_beta_schedule(self.config.n_timesteps)
        elif self.config.beta_schedule == "cosine":
            self.betas = cosine_beta_schedule(self.config.n_timesteps)
        elif self.config.beta_schedule == "vp":
            self.betas = vp_beta_schedule(self.config.n_timesteps)

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = jnp.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = jnp.concatenate(
            [jnp.ones(1), self.alphas_cumprod[:-1]]
        )

        # q(x_t | x_{t-1})
        self.sqrt_alphas_cumprod = jnp.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = jnp.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = jnp.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = jnp.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = jnp.sqrt(1.0 / self.alphas_cumprod - 1)

        # q(x_{t-1} | x_t, x_0)
        # see https://arxiv.org/pdf/2208.11970.pdf
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = jnp.log(
            jnp.clip(self.posterior_variance, a_min=1e-20)
        )
        self.posterior_mean_coef1 = (
            self.betas
            * jnp.sqrt(self.alphas_cumprod_prev)
            / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * jnp.sqrt(self.alphas)
            / (1.0 - self.alphas_cumprod)
        )

        self.noise_model = NoiseModel(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_size=self.config.hidden_size,
            time_embed_dim=self.config.time_embed_dim,
            w_init=w_init,
            b_init=b_init,
        )

    # ------------------------------------------ sampling ------------------------------------------#

    def predict_start_from_noise(self, x_t, t, noise):
        """
        if self.config.predict_epsilon, model output is (scaled) noise;
        otherwise, model predicts x0 directly
        """
        if self.config.predict_epsilon:
            return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
                - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        else:
            return noise

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, s):
        x_recon = self.predict_start_from_noise(x, t=t, noise=self.noise_model(x, t, s))

        if self.config.clip_denoised:
            jnp.clip(x_recon, -self.max_action, self.max_action)
        else:
            assert RuntimeError()

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t
        )
        return model_mean, posterior_variance, posterior_log_variance

    def p_sample(self, x, t, s):
        b, *_ = x.shape
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, s=s)

        noise = jax.random.normal(hk.next_rng_key(), shape=x.shape)

        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).astype(jnp.float32)).reshape(
            b, *((1,) * (len(x.shape) - 1))
        )
        return model_mean + nonzero_mask * jnp.exp(0.5 * model_log_variance) * noise

    def p_sample_loop(self, state, shape, return_diffusion=False):
        """
        sample from the inverse diffusion process, starting from x_t ~ N(0, I)
        """
        batch_size = shape[0]
        x = jax.random.normal(hk.next_rng_key(), shape=shape)

        if return_diffusion:
            diffusion = [x]

        for i in reversed(range(0, self.config.n_timesteps)):
            timesteps = jnp.full((batch_size,), i, dtype=jnp.int32)
            x = self.p_sample(x, timesteps, state)

            if return_diffusion:
                diffusion.append(x)

        if return_diffusion:
            return x, jnp.stack(diffusion, axis=1)
        else:
            return x

    def sample(self, state, *args, **kwargs):
        batch_size = state.shape[0]
        shape = (batch_size, self.action_dim)
        action = self.p_sample_loop(state, shape, *args, **kwargs)
        return jnp.clip(action, -self.max_action, self.max_action)

    # ------------------------------------------ training ------------------------------------------#

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = jax.random.normal(hk.next_rng_key(), shape=x_start.shape)

        sample = (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

        return sample

    def p_losses(self, x_start, state, t, weights=1.0):
        noise = jax.random.normal(hk.next_rng_key(), shape=x_start.shape)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        x_recon = self.model(x_noisy, t, state)

        assert noise.shape == x_recon.shape

        if self.config.predict_epsilon:
            loss = self.loss_fn(x_recon, noise, weights)
        else:
            loss = self.loss_fn(x_recon, x_start, weights)

        return loss

    def loss(self, x, state, weights=1.0):
        batch_size = len(x)
        t = jax.random.randint(
            hk.next_rng_key(), (batch_size,), 0, self.config.n_timesteps
        )
        return self.p_losses(x, state, t, weights)

    def __call__(self, state, *args, **kwargs):
        return self.sample(state, *args, **kwargs)
