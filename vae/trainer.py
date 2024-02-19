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

dist = tfp.distributions

eps = jnp.finfo(jnp.float32).eps.item()


class TrainingState(NamedTuple):
    params: hk.Params
    opt_state: optax.OptState


class Trainer:
    def __init__(self, config):
        self.rng_seq = hk.PRNGSequence(config.seed)

        # params
        self.env = gym.make(config.env_id)
        self.env.reset(seed=0)
        sample_obs = jnp.zeros((1, self.env.observation_space.shape[0]))

        # if discrete
        config = ml_collections.ConfigDict(config)
        if isinstance(self.env.action_space, gym.spaces.Discrete):
            self.is_discrete = True
            # unlock frozenconfigdict
            config.model.decoder.action_dim = int(self.env.action_space.n)
            config.model.policy.action_dim = int(self.env.action_space.n)
            config.model.decoder.is_discrete = True
            sample_action = jnp.zeros((1, 1))
        else:
            self.is_discrete = False
            config.model.decoder.is_discrete = False
            config.model.decoder.action_dim = int(self.env.action_space.shape[0])
            config.model.policy.action_dim = int(self.env.action_space.shape[0])
            sample_action = jnp.zeros((1, self.env.action_space.shape[0]))

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
            _, logits, log_prob = policy(obs)
            action = jnp.argmax(logits, axis=-1)
            return action

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
        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adam(config.lr),
        )
        all_params = (policy_params, value_fn_params)
        opt_state = optimizer.init(all_params)

        # create training state
        self.init_state = TrainingState(all_params, opt_state)
        self.train_state = self.init_state
        self.optimizer = optimizer

        orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        options = orbax.checkpoint.CheckpointManagerOptions(max_to_keep=4, create=True)
        self.cm = orbax.checkpoint.CheckpointManager(
            config.ckpt_dir, orbax_checkpointer, options
        )
        self.config = config

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
                entropy_weight,
            )
            updates, new_opt_state = optimizer.update(grad, state.opt_state)
            new_params = optax.apply_updates(state.params, updates)
            new_state = TrainingState(new_params, new_opt_state)
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

    def collect_rollout(self, train_state):
        states, actions, rewards = [], [], []
        obs, info = self.env.reset()
        done = False
        t = 0
        policy_params, _ = train_state.params
        while not done:
            rng_key = next(self.rng_seq)
            action = self._sample_action(policy_params, rng_key, obs)
            states.append(obs)
            actions.append(action)
            obs, reward, done, truncated, info = self.env.step(action.item())
            rewards.append(reward)
            t += 1
            done = done or truncated

        R = 0
        returns = []
        for r in rewards:
            R = r + self.config.discount * R
            returns.insert(0, R)

        returns = jnp.array(returns)

        # normalize returns
        # actually don't do this, this messes up the training
        # probably because we're already subtracting baseline?
        # returns = (returns - returns.mean()) / (returns.std() + eps)
        return (
            jnp.array(states),
            jnp.array(actions),
            jnp.array(rewards),
            jnp.array(returns),
        )

    def run_eval(self, train_state):
        eval_env = gym.make(self.config.env_id)
        returns = []
        policy_params, _ = train_state.params

        for _ in range(self.config.num_eval_episodes):
            obs, info = eval_env.reset()
            done = False
            total_reward = 0
            while not done:
                rng_key = next(self.rng_seq)
                action = self._sample_action(policy_params, rng_key, obs)
                obs, reward, done, truncated, _ = eval_env.step(action.item())
                total_reward += reward

                done = done or truncated
            returns.append(total_reward)
        print("Average return: ", jnp.mean(jnp.array(returns)))
