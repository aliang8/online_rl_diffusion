import jax
import optax
import haiku as hk
from model import Encoder, Decoder, VAE
from typing import NamedTuple
import jax.numpy as jnp
from functools import partial
from flax.training import train_state, checkpoints
import orbax
from tensorflow_probability.substrates import jax as tfp

dist = tfp.distributions


class TrainingState(NamedTuple):
    params: hk.Params
    opt_state: optax.OptState
    rng_key: jax.Array


class Trainer:
    def __init__(self, config):
        @hk.transform
        def model(x):
            vae = VAE(
                encoder=Encoder(**config.model.encoder),
                decoder=Decoder(**config.model.decoder),
            )
            return vae(x)

        @hk.transform
        def sample_vae(batch_size):
            vae = VAE(
                encoder=Encoder(**config.model.encoder),
                decoder=Decoder(**config.model.decoder),
            )
            return vae.sample(batch_size)

        init_rng_key = jax.random.PRNGKey(config.seed)

        # params
        sample_input = jnp.zeros((1, config.action_dim))  # dummy input, this is B, 1
        init_params = model.init(init_rng_key, sample_input)
        param_count = sum(x.size for x in jax.tree_leaves(init_params))
        print("number of params: ", param_count)

        # optimizer
        optimizer = optax.adam(config.lr)
        opt_state = optimizer.init(init_params)

        # create training state
        self.init_state = TrainingState(init_params, opt_state, init_rng_key)
        self.model = model
        self.optimizer = optimizer

        orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        options = orbax.checkpoint.CheckpointManagerOptions(max_to_keep=4, create=True)
        self.cm = orbax.checkpoint.CheckpointManager(
            config.ckpt_dir, orbax_checkpointer, options
        )

        def loss_fn(params, rng_key, input, entropy_weight):
            """
            ELBO loss: E_p[log(x)] - KL(d||q), where p ~ Be(0.5) and q ~ N(0,1).

            PG loss = -(r(s) * log(pi(a|s)) - H(pi(a|s))
            """
            # input shape is (B, 1)

            # run forward pass
            _, mean, var, recon = model.apply(params, rng_key, input)

            # compute MSE for reconstruction loss, this is log-likelihood
            recon_loss = jnp.square(input - recon).mean(axis=-1)
            # jax.debug.breakpoint()

            # compute KL divergence between unit Gaussian and Gaussian(mean, var)
            # scale is std
            p = dist.Normal(loc=jnp.zeros_like(mean), scale=jnp.ones_like(var))
            q = dist.Normal(loc=mean, scale=jnp.sqrt(var))
            kl_to_uniform_prior = dist.kl_divergence(q, p).sum(axis=-1)

            # ELBO is -likelihood - KL, want to maximize this term
            elbo = -recon_loss - kl_to_uniform_prior
            # elbo_loss = recon_loss + config.kl_weight * kl_to_uniform_prior

            # add entropy to encourage exploration and not get stuck at some local minima
            # elbo is proxy for log(p(a|s))
            # we want to explore actions that have low log(p(a|s)), we penalize if elbo is high
            r = self.reward(input).squeeze(-1)
            r_ent = (
                self.reward(input).squeeze(-1) * config.reward_weight
                - entropy_weight * elbo
            )
            pg_loss = -(r_ent * elbo).mean()

            metrics = {
                "recon_loss": recon_loss.mean(),
                "kl_to_uniform_prior": kl_to_uniform_prior.mean(),
                "elbo": elbo.mean(),
                "pg_loss": pg_loss,
                "reward": r.mean(),
            }
            return pg_loss, metrics

        self.loss_fn = loss_fn

        @jax.jit
        def update_step(state, input, entropy_weight):
            """
            input: batch of points
            returns: new TrainingState
            """
            rng_key, next_rng_key = jax.random.split(state.rng_key)
            (_, metrics), grad = jax.value_and_grad(self.loss_fn, has_aux=True)(
                state.params, rng_key, input, entropy_weight
            )
            updates, new_opt_state = optimizer.update(grad, state.opt_state)
            new_params = optax.apply_updates(state.params, updates)
            new_state = TrainingState(new_params, new_opt_state, next_rng_key)
            return new_state, metrics

        self._update_step = update_step

        @partial(jax.jit, static_argnums=(1,))
        def sample(state, batch_size):
            sample_rng, _ = jax.random.split(state.rng_key)
            return sample_vae.apply(state.params, sample_rng, batch_size)

        self._sample = sample

        @jax.jit
        def compute_logp(state, input):
            """
            logp is the elbo
            """
            model_rng, _ = jax.random.split(state.rng_key)
            _, mean, var, recon = model.apply(state.params, model_rng, input)

            recon_loss = jnp.square(input - recon).mean(axis=-1)
            # compute KL divergence between Gaussian(0,1) and Gaussian(mean, var)
            # scale is the std
            p = dist.Normal(loc=jnp.zeros_like(mean), scale=jnp.ones_like(var))
            q = dist.Normal(loc=mean, scale=jnp.sqrt(var))
            kl_to_uniform_prior = dist.kl_divergence(q, p).sum(axis=-1)

            # loss is the ELBO
            # ELBO is likelihood - KL
            # we want to maximize ELBO, so we minimize -likelihood + KL
            # we want to minimize this term.
            elbo = -recon_loss - kl_to_uniform_prior
            return elbo

        self._compute_logp = compute_logp

    def reward(self, x):
        # bandit reward function, a gaussian centered at 0
        # y1 = jnp.clip(-18 * (0.5 * x - 0.1) ** 2 + 1.1, a_min=0)
        # return y1
        y1 = jnp.clip(-18 * (2 * x + 1.2) ** 2 + 1.1, a_min=0)
        y2 = jnp.clip(-0.125 * (4 * x - 1.75) ** 2 + 0.5, a_min=0)
        # y2 = y2 * (x < 1.75 / 4).float()
        return y1 + y2

    def save_checkpoint(self, state, step, path="results"):
        self.cm.save(step, state)

    def load_checkpoint(self, path="results"):
        step = self.cm.latest_step()
        print(f"Loading checkpoint from step {step}")
        state = self.cm.restore_checkpoint(step)
        self.init_state = state
        return state
