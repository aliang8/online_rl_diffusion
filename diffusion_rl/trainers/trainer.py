"""
Train policy with RL algorithm
"""

from absl import app, logging
import jax
import optax
import jax.numpy as jnp
import numpy as np
import haiku as hk
import os
import tqdm
import pickle
import time
import flax
from ml_collections import ConfigDict, FieldReference, FrozenConfigDict, config_flags
from functools import partial
from flax.training.train_state import TrainState
from typing import Any
from tensorflow_probability.substrates import jax as tfp
import pickle
from pathlib import Path
import diffusion_rl.utils.utils as utils
from diffusion_rl.models.models import policy_fn, value_fn, actor_critic_fn
from ray import train, tune
from ray.train import RunConfig, ScalingConfig
import cv2

# import torch
import wandb
from flax import traverse_util
import gymnasium as gym
from collections import defaultdict as dd

eps = jnp.finfo(jnp.float32).eps


class BaseRLTrainer:
    def __init__(self, config: FrozenConfigDict):
        self.config = config
        self.rng_seq = hk.PRNGSequence(config.seed)

        # set torch seed to maintain reproducibility
        np.random.seed(config.seed)

        if self.config.use_wb:
            self.wandb_run = wandb.init(
                # set the wandb project where this run will be logged
                entity="glamor",
                project="online_rl_diffusion",
                name=config.exp_name,
                notes=self.config.notes,
                tags=self.config.tags,
                group=config.group_name if config.group_name else None,
                # track hyperparameters and run metadata
                config=self.config,
            )
        else:
            self.wandb_run = None

        # setup log dirs
        self.root_dir = Path(self.config.root_dir)
        self.ckpt_dir = self.root_dir / self.config.ckpt_dir
        logging.info(f"ckpt_dir: {self.ckpt_dir}")

        # make it
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.video_dir = self.root_dir / self.config.video_dir
        self.video_dir.mkdir(parents=True, exist_ok=True)

        self.log_dir = self.root_dir / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.jit_update_step = jax.jit(self.update_step)

        # create environment
        self.train_env = utils.make_env(self.config.env_id, seed=self.config.seed)
        self.obs_dim = self.train_env.observation_space.shape[0]
        self.is_continuous = isinstance(self.train_env.action_space, gym.spaces.Box)

        if self.is_continuous:
            self.action_dim = self.train_env.action_space.shape[0]
        else:
            self.action_dim = self.train_env.action_space.n

        print(f"max episode steps: {self.train_env.spec.max_episode_steps}")
        print(f"continuous action space: {self.is_continuous}")

        # create test environment
        self.test_env = utils.make_env(self.config.env_id, seed=self.config.seed + 100)

        self.ts = self.create_ts(next(self.rng_seq))

    def create_ts(self, rng):
        sample_obs = jnp.zeros((1, self.obs_dim))
        ac_params = actor_critic_fn.init(
            rng,
            sample_obs,
            hidden_size=self.config.hidden_size,
            action_dim=self.action_dim,
            share_backbone=self.config.share_backbone,
            is_continuous=self.is_continuous,
            policy_cls=self.config.policy,
            latent_dim=self.config.latent_dim,
        )

        partition_optimizers = {
            "policy": optax.adam(self.config.policy_lr),
            "value": optax.adam(self.config.critic_lr),
        }
        param_partitions = traverse_util.path_aware_map(
            lambda path, v: "value" if "value" in path[0] else "policy", ac_params
        )

        ac_opt = optax.multi_transform(partition_optimizers, param_partitions)

        param_count = sum(p.size for p in jax.tree_util.tree_leaves(ac_params))
        print(f"Number of policy parameters: {param_count}")

        ac_fn_apply = jax.tree_util.Partial(
            jax.jit(actor_critic_fn.apply, static_argnums=(3, 4, 5, 6, 7, 8, 9)),
            hidden_size=self.config.hidden_size,
            action_dim=self.action_dim,
            share_backbone=self.config.share_backbone,
            is_continuous=self.is_continuous,
            policy_cls=self.config.policy,
            latent_dim=self.config.latent_dim,
        )

        # self.eval_ac_fn_apply = jax.tree_util.Partial(
        #     jax.jit(actor_critic_fn.apply, static_argnums=(3, 4, 5, 6, 7, 8, 9)),
        #     hidden_size=self.config.hidden_size,
        #     action_dim=self.action_dim,
        #     share_backbone=self.config.share_backbone,
        #     is_continuous=self.is_continuous,
        #     policy_cls=self.config.policy,
        #     latent_dim=self.config.latent_dim,
        #     sample_prior=True,
        # )

        ts = TrainState.create(
            apply_fn=ac_fn_apply,
            params=ac_params,
            tx=ac_opt,
        )
        return ts

    def collect_single_rollout(self, train=True):
        # first step
        obs, _ = self.train_env.reset()

        # iterate
        states, actions, rewards, next_states, dones, frames = (
            [],
            [],
            [],
            [],
            [],
            [],
        )

        success = False
        done = False

        while not done:
            action_output, _ = self.ts.apply_fn(self.ts.params, next(self.rng_seq), obs)
            action = action_output.action

            if self.is_continuous:
                action *= self.config.action_scale

            # this is slow, need to convert to numpy
            next_obs, reward, done, truncated, info = self.train_env.step(
                np.array(action)
            )

            # save transition
            states.append(obs)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_obs)
            dones.append(done)

            # update state
            obs = next_obs

            # render
            if not train and self.config.save_eval_video:
                frame = self.train_env.render()
                frames.append(frame)

            # check if done
            if done:
                success = True
                break

            done = done or truncated

        # jax.debug.breakpoint()
        mask = np.ones_like(np.array(rewards))

        trajectory = utils.Trajectory(
            states=np.array(states),
            actions=np.array(actions),
            rewards=np.array(rewards),
            dones=np.array(dones),
            next_states=np.array(next_states),
            success=success,
            returns=None,
            mask=mask,
        )
        frames = np.array(frames)
        # compute discounted returns
        trajectory.returns = self.compute_discounted_returns(trajectory)

        # pad the first dimension to max_episode_steps
        if len(trajectory.rewards) < self.train_env.spec.max_episode_steps:
            for k, v in trajectory.__dict__.items():
                if isinstance(v, np.ndarray):
                    pad_length = self.train_env.spec.max_episode_steps - v.shape[0]
                    if len(v.shape) == 1:
                        v = np.pad(v, ((0, pad_length)))
                    else:
                        v = np.pad(v, ((0, pad_length), (0, 0)))

                    setattr(trajectory, k, v)

        # for k, v in trajectory.items():
        #     if isinstance(v, np.ndarray):
        #         print(k, v.shape)

        return trajectory, frames

    def run_eval_rollouts(self, epoch):
        rollout_metrics = dd(list)
        all_frames = []

        total_rollout_time = 0
        for eval_episode_indx in range(self.config.num_eval_episodes):
            trajectory, frames = self.collect_single_rollout(train=False)

            metrics = {
                "return": trajectory.rewards.sum(),
                "success": np.array(trajectory.success),
                "episode_length": np.array(sum(trajectory.mask)),
            }

            for k, v in metrics.items():
                rollout_metrics[k].append(v)

            # save videos
            if (
                self.config.save_eval_video
                and eval_episode_indx < self.config.num_eval_video_save
            ):
                # write some text on the frames
                y0, dy = 50, 20
                for timestep, image in enumerate(frames):
                    text = f"epoch: {epoch} \ntimestep: {timestep} \nreturn: {round(np.sum(trajectory.rewards[:timestep+1]),2)}\nstate: {np.round(trajectory.states[timestep], 2)}\naction: {np.round(trajectory.actions[timestep],2)}\ndone: {trajectory.dones[timestep]}\nsuccess: {trajectory.success}"

                    for i, line in enumerate(text.split("\n")):
                        y = y0 + i * dy
                        frames[timestep] = cv2.putText(
                            frames[timestep],
                            line,
                            (25, y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.3,
                            (0, 0, 255),
                            1,
                            cv2.LINE_AA,
                        )

                all_frames.append(frames)

        # save video
        if self.config.save_eval_video and self.wandb_run is not None:
            # N x T x H x W x C
            all_frames = np.array(all_frames)
            all_frames = all_frames[:5]

            # make N x T x C x H x W
            all_frames = np.transpose(all_frames, (0, 1, 4, 2, 3))
            self.wandb_run.log(
                {f"rollout": wandb.Video(all_frames, fps=30, format="mp4")}
            )
            self.wandb_run.log(
                {
                    "time/rollout_time": total_rollout_time,
                    "time/avg_rollout_time": total_rollout_time
                    / self.config.num_eval_episodes,
                }
            )

        rollout_metrics = {k: np.mean(v) for k, v in rollout_metrics.items()}
        return rollout_metrics

    def policy_loss_fn(self, params, ts, rng_key, trajectory):
        logging.info("inside policy loss")

        # compute reinforce loss
        action_output, value_estimate = ts.apply_fn(params, rng_key, trajectory.states)
        action_dist = action_output.action_dist

        # compute action log probabilities
        a_log_probs = action_dist.log_prob(trajectory.actions).squeeze()

        # jax.debug.breakpoint()

        if self.config.baseline:
            advantage = trajectory.returns - value_estimate
        else:
            returns = trajectory.returns
            # take mean of masked returns
            returns_mean = jnp.mean(returns, where=trajectory.mask)
            returns_std = jnp.std(returns, where=trajectory.mask)
            # apply whitening to returns
            returns = (returns - returns_mean) / (returns_std + eps)
            advantage = returns

        pg_loss = -a_log_probs * advantage
        pg_loss *= trajectory.mask

        # jax.debug.breakpoint()

        pg_loss = pg_loss.sum()

        # add exploration bonus, want to maximize entropy
        entropy_loss = self.config.entropy_weight * action_dist.entropy().sum()

        total_loss = pg_loss - entropy_loss
        metrics = {"pg_loss": pg_loss, "entropy_loss": entropy_loss}

        if self.config.baseline:
            # compute value loss
            value_loss = (value_estimate - returns) ** 2
            value_loss *= trajectory.mask
            value_loss = value_loss.sum()
            total_loss += self.config.value_coeff * value_loss
            metrics["value_loss"] = value_loss

        return total_loss, metrics

    def update_step(self, ts, trajectory, rng_key):
        (policy_loss, metrics), grads = jax.value_and_grad(
            jax.jit(self.policy_loss_fn), has_aux=True
        )(ts.params, ts, rng_key, trajectory)
        ts = ts.apply_gradients(grads=grads)
        return ts, policy_loss, metrics

    def compute_discounted_returns(self, trajectory):
        # compute discounted returns
        rewards = trajectory.rewards
        dones = trajectory.dones
        returns = np.zeros_like(rewards)
        R = 0
        for t in reversed(range(len(rewards))):
            R = rewards[t] + self.config.gamma * R * (1 - dones[t])
            returns[t] = R
        return returns

    def train(self):
        # main training loop for on-policy learning

        # Iterate over number of episodes
        for episode_indx in tqdm.tqdm(range(self.config.num_training_episodes)):
            # collect rollout of transitions
            rollout_time = time.time()
            trajectory, _ = self.collect_single_rollout()
            rollout_time = time.time() - rollout_time

            # update policy
            self.ts, loss, train_metrics = self.jit_update_step(
                self.ts, trajectory, next(self.rng_seq)
            )

            train_metrics.update(
                {
                    "return": trajectory.rewards.sum(),
                    "success": np.array(trajectory.success).astype(float),
                    "episode_length": np.array(sum(trajectory.mask)),
                    "rollout_time": rollout_time,
                }
            )

            if self.wandb_run is not None:
                train_metrics = {f"train/{k}": v for k, v in train_metrics.items()}
                self.wandb_run.log(train_metrics)

            # run evaluate
            if self.config.eval_every and episode_indx % self.config.eval_every == 0:
                eval_metrics = self.run_eval_rollouts(epoch=episode_indx)

                if self.wandb_run is not None:
                    eval_metrics = {f"eval/{k}": v for k, v in eval_metrics.items()}
                    self.wandb_run.log(eval_metrics)

            # save policy
            if self.config.save_every and episode_indx % self.config.save_every == 0:
                ckpt_file = self.ckpt_dir / f"policy_{episode_indx}.pkl"
                with open(ckpt_file, "wb") as f:
                    pickle.dump(self.ts.params, f)
