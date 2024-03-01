"""
Train policy with RL algorithm

Usage:
python3 trainer.py \
    --config=configs/rl_config.py \
    --config.mode=train \
"""

from absl import app
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
import mlogger
from tensorflow_probability.substrates import jax as tfp
import pickle
from pathlib import Path
import utils
from models import policy_fn
from ray import train, tune
from ray.train import RunConfig, ScalingConfig
import cv2


os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.01"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

_CONFIG = config_flags.DEFINE_config_file("config")
eps = jnp.finfo(jnp.float32).eps.item()


class BaseRLTrainer:
    def __init__(self, config: FrozenConfigDict):
        self.config = config
        self.rng_seq = hk.PRNGSequence(config.seed)

        # logger
        self.plotter = mlogger.VisdomPlotter(
            {
                "env": self.config.vizdom_name,
                "server": "http://localhost",
                "port": 8097,
            },
            manual_update=True,
        )

        self.xp = mlogger.Container()
        self.xp.config = mlogger.Config(plotter=self.plotter)
        self.xp.config.update(**self.config)
        self.xp.train = mlogger.Container()
        self.xp.test = mlogger.Container()

        # setup log dirs
        self.ckpt_dir = Path(self.config.root_dir) / self.config.results_dir
        # make it
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.video_dir = Path(self.config.root_dir) / self.config.video_dir
        self.video_dir.mkdir(parents=True, exist_ok=True)

        self.loss_keys = [
            ("pg_loss", mlogger.metric.Average, "PG Loss"),
            ("entropy_loss", mlogger.metric.Average, "Entropy Loss"),
            ("return", mlogger.metric.Average, "Returns"),
            ("success", mlogger.metric.Average, "Success Rate"),
            ("episode_length", mlogger.metric.Average, "Episode Length"),
        ]
        print("loss keys: ", self.loss_keys)
        for lk, logger_cls, title in self.loss_keys:
            self.xp.train.__setattr__(
                lk,
                logger_cls(plotter=self.plotter, plot_title=title, plot_legend="train"),
            )
            self.xp.test.__setattr__(
                lk,
                logger_cls(plotter=self.plotter, plot_title=title, plot_legend="test"),
            )

        self.jit_update_step = jax.jit(self.update_step)

        # create environment
        self.train_env = utils.make_env(self.config.env_name, seed=self.config.seed)
        self.obs_dim = self.train_env.observation_space.shape[0]
        self.action_dim = self.train_env.action_space.shape[0]

        # create test environment
        self.test_env = utils.make_env(
            self.config.env_name, seed=self.config.seed + 100
        )

        self.ts = self.create_ts(next(self.rng_seq))

    def create_ts(self, rng):
        sample_obs = jnp.zeros((1, self.obs_dim))
        policy_params = policy_fn.init(
            rng,
            sample_obs,
            hidden_size=self.config.hidden_size,
            action_dim=self.action_dim,
        )

        policy_opt = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adam(self.config.policy_lr),
        )

        param_count = sum(p.size for p in jax.tree_util.tree_leaves(policy_params))
        print(f"Number of policy parameters: {param_count}")

        policy_fn_apply = partial(
            jax.jit(policy_fn.apply, static_argnums=(3, 4)),
            hidden_size=self.config.hidden_size,
            action_dim=self.action_dim,
        )

        ts = TrainState.create(
            apply_fn=policy_fn_apply,
            params=policy_params,
            tx=policy_opt,
        )
        return ts

    def collect_single_rollout(self, train=True):
        # first step
        obs, _ = self.train_env.reset()

        # iterate
        states, actions, rewards, next_states, dones, frames = [], [], [], [], [], []

        success = False
        for step in range(self.config.max_episode_steps):
            rng_key = next(self.rng_seq)
            action, action_dist, log_prob = self.ts.apply_fn(
                self.ts.params, rng_key, obs
            )

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

        trajectory = utils.Trajectory(
            states=np.array(states),
            actions=np.array(actions),
            rewards=np.array(rewards),
            dones=np.array(dones),
            next_states=np.array(next_states),
            success=success,
        )
        frames = np.array(frames)
        return trajectory, frames

    def run_eval_rollouts(self, epoch):
        # reset metrics
        for metric in self.xp.test.metrics():
            metric.reset()

        for eval_episode_indx in range(self.config.num_eval_episodes):
            trajectory, frames = self.collect_single_rollout(train=False)
            metrics = {
                "return": trajectory.rewards.sum(),
                "success": np.array(trajectory.success),
                "episode_length": np.array(len(trajectory.rewards)),
            }

            # save videos
            if (
                self.config.save_eval_video
                and eval_episode_indx < self.config.num_eval_video_save
            ):
                # video_file = self.video_dir / f"eval_{eval_episode_indx}.mp4"
                # print(f"Saving video to: {video_file}")
                # # save frames
                # fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                # out = cv2.VideoWriter(
                #     str(video_file), fourcc, 20.0, (frames.shape[2], frames.shape[1])
                # )
                # for frame in frames:
                #     out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                # out.release()
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

                self.plotter.viz.video(
                    tensor=frames,
                    env=self.plotter.viz.env,
                    win=f"eval_video_{eval_episode_indx}_{epoch}",
                )
                # self.plotter.viz.matplot(plt, env=self.plotter.viz.env)
            # log metrics
            for lk in metrics.keys():
                self.xp.test.__getattribute__(lk).update(
                    metrics[lk].item(), weighting=self.config.num_eval_episodes
                )

        for metric in self.xp.test.metrics():
            metric.log()

        return

    def policy_loss_fn(self, params, ts, rng_key, trajectory, returns):
        # compute reinforce loss
        _, action_dist, _ = ts.apply_fn(params, rng_key, trajectory.states)

        # compute action log probabilities
        a_log_probs = action_dist.log_prob(trajectory.actions).squeeze(axis=-1)

        # apply whitening to returns
        returns = (returns - jnp.mean(returns)) / (jnp.std(returns) + eps)
        # advantage = jax.lax.stop_gradient(returns - value_estimate)
        advantage = returns
        pg_loss = -a_log_probs * advantage
        pg_loss = pg_loss.sum()

        # add exploration bonus, want to maximize entropy
        entropy_loss = self.config.entropy_weight * action_dist.entropy().sum()

        total_loss = pg_loss - entropy_loss

        metrics = {"pg_loss": pg_loss, "entropy_loss": entropy_loss}

        return total_loss, metrics

    def update_step(self, ts, trajectory, returns, rng_key):
        (policy_loss, metrics), grads = jax.value_and_grad(
            self.policy_loss_fn, has_aux=True
        )(ts.params, ts, rng_key, trajectory, returns)
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
            for metric in self.xp.train.metrics():
                metric.reset()

            # collect rollout of transitions
            trajectory, _ = self.collect_single_rollout()

            # compute discounted returns
            returns = self.compute_discounted_returns(trajectory)

            # update policy
            self.ts, loss, metrics = self.jit_update_step(
                self.ts, trajectory, returns, next(self.rng_seq)
            )

            metrics.update(
                {
                    "return": trajectory.rewards.sum(),
                    "success": np.array(trajectory.success).astype(float),
                    "episode_length": np.array(len(trajectory.rewards)),
                }
            )

            # log metrics
            for lk in metrics.keys():
                self.xp.train.__getattribute__(lk).update(
                    metrics[lk].item(), weighting=1
                )

            # run evaluate
            if self.config.eval_every and episode_indx % self.config.eval_every == 0:
                self.run_eval_rollouts(epoch=episode_indx)

            # save policy
            if self.config.save_every and episode_indx % self.config.save_every == 0:
                ckpt_file = self.ckpt_dir / "policy_params.pkl"
                with open(ckpt_file, "wb") as f:
                    pickle.dump(self.ts.params, f)

            for metric in self.xp.train.metrics():
                metric.log()

            # update plots
            self.plotter.update_plots()


def train_model_fn(config):
    trial_dir = train.get_context().get_trial_dir()
    if trial_dir:
        print("Trial dir: ", trial_dir)
        config["root_dir"] = Path(trial_dir)
        base_name = Path(trial_dir).name
    else:
        base_name = "base"
    config["vizdom_name"] = config["ray_experiment_name"] + "_" + base_name

    # wrap config in ConfigDict
    config = ConfigDict(config)

    print(config)
    trainer = BaseRLTrainer(config)
    if config.mode == "train":
        trainer.train()
    elif config.mode == "eval":
        trainer.eval()


def main(_):
    config = _CONFIG.value.to_dict()
    if config["smoke_test"] is False:
        # run with ray tune
        param_space = {
            "seed": tune.grid_search([0, 1]),
            "policy_lr": tune.grid_search([3e-4, 1e-3, 1e-4]),
        }
        config.update(param_space)
        train_model = tune.with_resources(train_model_fn, {"cpu": 2, "gpu": 1})

        run_config = RunConfig(
            name=config["ray_experiment_name"],
            local_dir="/scr/aliang80/online_rl_diffusion/ray_results",
            storage_path="/scr/aliang80/online_rl_diffusion/ray_results",
            log_to_file=True,
        )
        tuner = tune.Tuner(train_model, param_space=config, run_config=run_config)
        results = tuner.fit()
        print(results)
    else:
        # run without ray tune
        train_model_fn(config)


if __name__ == "__main__":
    app.run(main)
