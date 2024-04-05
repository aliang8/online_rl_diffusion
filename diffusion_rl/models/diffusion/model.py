import copy
import numpy as np
import jax
import optax
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
        output_dim: int,
        hidden_size: int,
        time_embed_dim: int,
        conditional: bool,
        w_init: hk.initializers.Initializer,
        b_init: hk.initializers.Initializer,
    ):
        super().__init__()
        self.conditional = conditional
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
                hk.Linear(output_dim, **init_kwargs),
            ]
        )

    def __call__(self, x, time, cond=None):
        t = self.time_mlp(time)
        if self.conditional and cond is not None:
            x = jnp.concatenate([x, t, cond], axis=1)
        else:
            x = jnp.concatenate([x, t], axis=1)
        x = self.net(x)
        return x


class DiffusionDDPM(hk.Module):
    def __init__(
        self,
        config: config_dict.ConfigDict,
        output_dim: int,
        w_init: hk.initializers.Initializer = hk.initializers.VarianceScaling(2.0),
        b_init: hk.initializers.Initializer = hk.initializers.Constant(0.0),
    ):
        super().__init__()

        self.config = config
        self.output_dim = output_dim

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
            output_dim=self.output_dim,
            hidden_size=self.config.hidden_size,
            time_embed_dim=self.config.time_embed_dim,
            conditional=self.config.conditional,
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

    def p_mean_variance(self, x, t, cond=None):
        x_recon = self.predict_start_from_noise(
            x, t=t, noise=self.noise_model(x, t, cond)
        )
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t
        )
        return model_mean, posterior_variance, posterior_log_variance

    def p_sample(self, x, t, cond=None):
        b, *_ = x.shape
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, cond=cond)

        noise = jax.random.normal(hk.next_rng_key(), shape=x.shape)

        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).astype(jnp.float32)).reshape(
            b, *((1,) * (len(x.shape) - 1))
        )
        return model_mean + nonzero_mask * jnp.exp(0.5 * model_log_variance) * noise

    def __call__(self, batch_size=None, cond=None, return_diffusion=False):
        """
        sample from the inverse diffusion process, starting from x_t ~ N(0, I)
        """
        if cond is not None:
            batch_size = cond.shape[0]
        else:
            assert batch_size is not None

        x = jax.random.normal(hk.next_rng_key(), shape=(batch_size, self.output_dim))

        if return_diffusion:
            diffusion = [x]

        for i in reversed(range(0, self.config.n_timesteps)):
            timesteps = jnp.full((batch_size,), i, dtype=jnp.int32)
            x = self.p_sample(x, timesteps, cond)

            if return_diffusion:
                diffusion.append(x)

        if return_diffusion:
            return x, jnp.stack(diffusion, axis=1)
        else:
            return x

    # ------------------------------------------ training ------------------------------------------#

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = jax.random.normal(hk.next_rng_key(), shape=x_start.shape)

        sample = (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

        return sample

    def loss(self, x, cond=None):
        batch_size = len(x)
        t = jax.random.randint(
            hk.next_rng_key(), (batch_size,), 0, self.config.n_timesteps
        )
        noise = jax.random.normal(hk.next_rng_key(), shape=x.shape)

        x_noisy = self.q_sample(x_start=x, t=t, noise=noise)
        x_recon = self.noise_model(x_noisy, t, cond)

        assert noise.shape == x_recon.shape

        if self.config.predict_epsilon:
            loss = optax.squared_error(x_recon, noise)
        else:
            loss = optax.squared_error(x_recon, x)
        return loss
