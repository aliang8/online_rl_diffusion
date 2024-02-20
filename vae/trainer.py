import jax
import optax
import haiku as hk
from model import (
    Encoder,
    Decoder,
    VAE,
    StandardPrior,
    FlowPrior,
    StateConditionedPrior,
    ValueFunction,
    Policy,
)
from typing import NamedTuple
import jax.numpy as jnp
from functools import partial
from flax.training import train_state, checkpoints
import orbax
from tensorflow_probability.substrates import jax as tfp
import gymnasium as gym
import ml_collections
import time
from loss import policy_loss_fn, vae_policy_loss_fn
import gymnax

dist = tfp.distributions

eps = jnp.finfo(jnp.float32).eps.item()


def create_learning_rate_fn(
    num_epochs, warmup_epochs, base_learning_rate, steps_per_epoch
):
    """Creates learning rate schedule."""
    warmup_fn = optax.linear_schedule(
        init_value=0.0,
        end_value=base_learning_rate,
        transition_steps=warmup_epochs * steps_per_epoch,
    )
    cosine_epochs = max(num_epochs - warmup_epochs, 1)
    cosine_fn = optax.cosine_decay_schedule(
        init_value=base_learning_rate, decay_steps=cosine_epochs * steps_per_epoch
    )
    schedule_fn = optax.join_schedules(
        schedules=[warmup_fn, cosine_fn],
        boundaries=[warmup_epochs * steps_per_epoch],
    )
    return schedule_fn


class TrainingState(NamedTuple):
    params: hk.Params
    policy_opt_state: optax.OptState
    vf_opt_state: optax.OptState
    step: int


class Trainer:
    def __init__(self, config):
        self.rng_seq = hk.PRNGSequence(config.seed)

        # params
        # self.env = gym.make(config.env_id)
        # self.env.reset(seed=0)
        self.env, self.env_params = gymnax.make(config.env_id)
        self.env.reset(next(self.rng_seq), self.env_params)
        obs_space = self.env.observation_space(self.env_params)
        action_space = self.env.action_space(self.env_params)
        sample_obs = jnp.zeros((1, obs_space.shape[0]))

        # if discrete
        config = ml_collections.ConfigDict(config)
        if isinstance(action_space, gymnax.environments.spaces.Discrete):
            self.is_discrete = True
            # unlock frozenconfigdict
            config.model.decoder.action_dim = int(action_space.n)
            config.model.policy.action_dim = int(action_space.n)
            config.model.decoder.is_discrete = True
            sample_action = jnp.zeros((1, 1))
        else:
            self.is_discrete = False
            config.model.decoder.is_discrete = False
            config.model.decoder.action_dim = int(action_space.shape[0])
            config.model.policy.action_dim = int(action_space.shape[0])
            sample_action = jnp.zeros((1, action_space.shape[0]))

        @hk.transform
        def vae_policy_forward(obs, action):
            if config.prior_type == "standard":
                prior = StandardPrior(**config.model.standard)
            elif config.prior_type == "flow":
                prior = FlowPrior(**config.model.flow)
            elif config.prior_type == "state_conditioned":
                prior = StateConditionedPrior(**config.model.state_conditioned)
            else:
                raise ValueError("Invalid prior type")

            vae = VAE(
                encoder=Encoder(**config.model.encoder),
                decoder=Decoder(**config.model.decoder),
                prior=prior,
            )
            return vae(obs, action)

        @hk.transform
        def policy_forward(obs):
            policy = Policy(**config.model.policy)
            return policy(obs)

        @hk.transform
        def sample_vae(batch_size):
            if config.prior_type == "standard":
                prior = StandardPrior(**config.model.standard)
            elif config.prior_type == "flow":
                prior = FlowPrior(**config.model.flow)
            elif config.prior_type == "state_conditioned":
                prior = StateConditionedPrior(**config.model.state_conditioned)
            else:
                raise ValueError("Invalid prior type")
            vae = VAE(
                encoder=Encoder(**config.model.encoder),
                decoder=Decoder(**config.model.decoder),
                prior=prior,
            )
            decoder_output = vae.sample(batch_size)
            if self.is_discrete:
                action = jax.random.categorical(
                    hk.next_rng_key(), logits=decoder_output
                )
            else:
                action = decoder_output
            return action

        @hk.transform
        def sample_policy(obs):
            policy = Policy(**config.model.policy)
            action, logits, log_prob = policy(obs)
            return action, log_prob

        @hk.transform
        def value_fn(obs):
            value_net = ValueFunction(**config.model.value_function)
            value_estimate = value_net(obs)
            return value_estimate

        config.lock()

        if config.policy_cls == "vae":
            self._sample_action = jax.jit(sample_vae.apply)
            policy_params = vae_policy_forward.init(
                next(self.rng_seq), sample_obs, sample_action
            )
            loss_fn = vae_policy_loss_fn
            policy = vae_policy_forward
        else:
            self._sample_action = jax.jit(sample_policy.apply)
            policy_params = policy_forward.init(next(self.rng_seq), sample_obs)
            loss_fn = policy_loss_fn
            policy = policy_forward

        value_fn_params = value_fn.init(next(self.rng_seq), sample_obs)
        param_count = sum(x.size for x in jax.tree_leaves(policy_params))
        print("number of params: ", param_count)
        value_fn_param_count = sum(x.size for x in jax.tree_leaves(value_fn_params))
        print("number of value function params: ", value_fn_param_count)

        # optimizer
        self.lr_fn = create_learning_rate_fn(
            num_epochs=config.num_epochs,
            warmup_epochs=100,
            base_learning_rate=config.lr,
            steps_per_epoch=1,
        )

        if config.use_lr_scheduler:
            optimizer = optax.chain(
                optax.clip_by_global_norm(1.0),
                optax.adam(self.lr_fn),
            )
        else:
            optimizer = optax.chain(
                optax.clip_by_global_norm(1.0),
                optax.adam(config.lr),
            )
            vf_optimizer = optax.chain(
                optax.clip_by_global_norm(1.0),
                optax.adam(config.vf_lr),
            )
        policy_opt_state = optimizer.init(policy_params)
        vf_opt_state = vf_optimizer.init(value_fn_params)

        # create training state
        self.init_state = TrainingState(
            (policy_params, value_fn_params), policy_opt_state, vf_opt_state, 0
        )
        self.train_state = self.init_state
        self.optimizer = optimizer

        orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        options = orbax.checkpoint.CheckpointManagerOptions(max_to_keep=4, create=True)
        self.cm = orbax.checkpoint.CheckpointManager(
            config.ckpt_dir, orbax_checkpointer, options
        )
        self.config = config

        # @partial(jax.jit, static_argnames=("entropy_weight"))
        @jax.jit
        def update_step(
            state,
            rng_key,
            obss,
            actions,
            rewards,
            dones,
            returns,
            entropy_weight,
        ):
            """
            input: batch of points
            returns: new TrainingState
            """
            if self.config.policy_cls == "vae":
                extra_kwargs = dict(
                    prior_loss_weight=config.prior_loss_weight,
                    kl_weight=config.kl_weight,
                    is_discrete=self.is_discrete,
                    entropy_weight=entropy_weight,
                )
            else:
                extra_kwargs = dict()
            (_, metrics), grad = jax.value_and_grad(loss_fn, has_aux=True)(
                state.params,
                rng_key,
                value_fn,
                policy,
                obss,
                actions,
                rewards,
                dones,
                returns,
                **extra_kwargs,
            )
            policy_params, value_fn_params = state.params
            updates, new_policy_opt_state = optimizer.update(
                grad[0], state.policy_opt_state
            )
            new_policy_params = optax.apply_updates(policy_params, updates)
            updates, new_vf_opt_state = vf_optimizer.update(grad[1], state.vf_opt_state)
            new_vf_params = optax.apply_updates(value_fn_params, updates)
            new_state = TrainingState(
                (new_policy_params, new_vf_params),
                new_policy_opt_state,
                new_vf_opt_state,
                state.step + 1,
            )
            return new_state, metrics

        self._update_step = update_step

    def save_checkpoint(self, state, step, path="results"):
        self.cm.save(step, state)

    def load_checkpoint(self, path="results"):
        step = self.cm.latest_step()
        print(f"Loading checkpoint from step {step}")
        state = self.cm.restore_checkpoint(step)
        self.init_state = state
        return state

    def collect_rollout(self, train_state, rng_key):
        """Rollout a jitted gymnax episode with lax.scan."""
        policy_params, value_fn_params = train_state.params
        reset_rng, step_rng = jax.random.split(rng_key)
        # Reset the environment
        obs, state = self.env.reset(reset_rng, self.env_params)

        def policy_step(state_input, tmp):
            """lax.scan compatible step transition in jax env."""
            obs, state, policy_params, rng = state_input
            rng_step, rng_sample, rng = jax.random.split(rng, 3)
            action, _ = self._sample_action(policy_params, rng_sample, obs)
            next_obs, next_state, reward, done, _ = self.env.step(
                rng_step, state, action, self.env_params
            )
            carry = [next_obs, next_state, policy_params, rng]
            return carry, [obs, action, reward, next_obs, done]

        # Scan over episode step loop
        _, scan_out = jax.lax.scan(
            policy_step, [obs, state, policy_params, step_rng], (), 500
        )
        # Return masked sum of rewards accumulated by agent in episode
        obs, action, reward, next_obs, done = scan_out

        dones_mask = jnp.logical_not(jnp.cumsum(done, axis=0).astype(bool))
        reward *= dones_mask

        # compute returns
        def scan_fn(prev_return, reward):
            return reward + 1.0 * prev_return, reward + 1.0 * prev_return

        init_return = jnp.zeros_like(reward[0])
        _, returns = jax.lax.scan(scan_fn, init_return, reward[::-1])
        returns = returns[::-1]
        return obs, action, returns, next_obs, dones_mask

    # def run_eval(self, train_state):
    #     # eval_env = gym.make(self.config.env_id)
    #     eval_env, _ = gymnax.make(self.config.env_id)
    #     returns = []
    #     policy_params, _ = train_state.params

    #     for _ in range(self.config.num_eval_episodes):
    #         obs, state = eval_env.reset(next(self.rng_seq), self.env_params)
    #         done = False
    #         total_reward = 0
    #         while not done:
    #             rng_key = next(self.rng_seq)
    #             action, _ = self._sample_action(policy_params, rng_key, obs)
    #             obs, state, reward, done, _ = eval_env.step(
    #                 next(self.rng_seq), state, action, self.env_params
    #             )
    #             total_reward += reward

    #         returns.append(total_reward)
    #     print("Average return: ", jnp.mean(jnp.array(returns)))
