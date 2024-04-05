from absl import logging
import jax
import time
import pickle
from ml_collections import ConfigDict
import numpy.random as npr
from pathlib import Path
import jax.tree_util as jtu
import jax.numpy as jnp
import haiku as hk
import optax
import functools
import gymnasium as gym
import tqdm
import d4rl
import gymnasium as gym
from collections import defaultdict as dd
from flax.training.train_state import TrainState
from ml_collections.config_dict import ConfigDict, FrozenConfigDict
from diffusion_rl.trainers.base_trainer import BaseTrainer
import diffusion_rl.utils.general_utils as gutl
from diffusion_rl.models.diffusion.helpers import init_params as init_params_diffusion
from diffusion_rl.models.diffusion.helpers import (
    diffusion_apply_fn,
    diffusion_compute_loss_fn,
)
from diffusion_rl.utils.rollout import run_rollouts
from diffusion_rl.utils.data import load_dataset


def create_ts(
    config: ConfigDict,
    rng: jax.random.PRNGKey,
    input_dim: int,
    output_dim: int,
):
    if config.load_from_ckpt:
        logging.info("loading model from checkpoint")
        model_ckpt_dir = Path(config.root_dir) / config.model_ckpt_dir
        ckpt_file = model_ckpt_dir / f"ckpt_{config.checkpoint_step}.pkl"

        with open(ckpt_file, "rb") as f:
            ckpt = pickle.load(f)
            params = ckpt["ts_policy"]
    else:
        if config.policy.name == "diffusion":
            params = init_params_diffusion(config.policy, rng, input_dim, output_dim)

    if config.policy.name == "diffusion":
        policy_fn = diffusion_apply_fn

    num_params = sum(p.size for p in jax.tree_util.tree_leaves(params))
    logging.info(f"num params = {num_params}")

    tx = optax.chain(
        # optax.clip(config.vae.max_grad_norm),
        optax.adam(config.lr, eps=config.eps),
    )
    policy_apply = functools.partial(
        jax.jit(
            policy_fn.apply,
            static_argnames=("config", "output_dim", "batch_size"),
        ),
        config=FrozenConfigDict(config.policy),
        output_dim=output_dim,
    )
    ts = TrainState.create(
        apply_fn=policy_apply,
        params=params,
        tx=tx,
    )
    return ts


class OfflineTrainer(BaseTrainer):
    def __init__(self, config: ConfigDict):
        super().__init__(config)

        # load dataset
        self.dataset = load_dataset(config, self.envs)
        logging.info(
            f"loaded dataset, observation shape: {self.dataset.observations.shape}, action shape: {self.dataset.actions.shape}"
        )

        self.state_dim = self.dataset.observations.shape[-1]
        self.action_dim = self.dataset.actions.shape[-1]

        self.ts_policy = create_ts(
            config,
            next(self.rng_seq),
            input_dim=self.state_dim,
            output_dim=self.action_dim,
        )

        self.compute_loss = functools.partial(
            jax.jit(
                diffusion_compute_loss_fn.apply,
                static_argnames=("config", "output_dim"),
            ),
            config=FrozenConfigDict(config.policy),
            output_dim=self.action_dim,
        )

        def loss_fn(params, ts, batch, rng):
            action_preds = ts.apply_fn(
                params,
                rng,
                cond=batch.observations.astype(jnp.float32),
            )

            if self.continuous_actions:
                # compute MSE loss
                loss = optax.squared_error(action_preds, batch.actions)
            else:
                # compute cross entropy with logits
                loss = optax.softmax_cross_entropy_with_integer_labels(
                    action_preds, batch.actions.squeeze(axis=-1).astype(jnp.int32)
                )
            # loss = self.compute_loss(
            #     params, rng, states=batch.observations, actions=batch.actions
            # )
            loss = jnp.mean(loss)
            metrics = {"bc_loss": loss}

            return loss, metrics

        def update_step(ts, batch, rng):
            (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(
                ts.params, ts, batch, rng
            )
            ts = ts.apply_gradients(grads=grads)
            return ts, metrics

        self.jit_update_step = jax.jit(update_step)

    def train(self):
        # first eval
        if not self.config.skip_first_eval:
            eval_metrics = self.eval()
            if self.wandb_run is not None:
                metrics = gutl.prefix_dict_keys(eval_metrics, prefix="eval/")
                self.wandb_run.log(metrics)

        logging.info("start training")

        # train
        for epoch in tqdm.tqdm(range(self.config.num_epochs)):
            # iterate over batches of data
            start_time = time.time()
            epoch_metrics = dd(list)
            for _ in range(self.config.num_iter_per_epoch):
                self.ts_policy, metrics = self.jit_update_step(
                    self.ts_policy,
                    self.dataset.sample(self.config.batch_size),
                    next(self.rng_seq),
                )
                for k, v in metrics.items():
                    epoch_metrics[k].append(v)

            # average a list of dicts using jax tree operations
            for k, v in epoch_metrics.items():
                epoch_metrics[k] = jnp.mean(jnp.array(v))

            epoch_time = time.time() - start_time
            epoch_metrics["time/epoch"] = epoch_time

            if self.wandb_run is not None:
                metrics = gutl.prefix_dict_keys(epoch_metrics, prefix="train/")
                self.wandb_run.log(metrics)

            if (epoch + 1) % self.config.eval_interval == 0:
                logging.info("running evaluation")
                eval_metrics = self.eval()
                if self.wandb_run is not None:
                    eval_metrics = gutl.prefix_dict_keys(eval_metrics, prefix="eval/")
                    self.wandb_run.log(eval_metrics)

            if (epoch + 1) % self.config.save_interval == 0:
                # save to pickle
                ckpt_file = Path(self.ckpt_dir) / f"ckpt_{epoch + 1}.pkl"
                logging.debug(f"saving checkpoint to {ckpt_file}")
                with open(ckpt_file, "wb") as f:
                    pickle.dump(
                        {
                            "config": self.config.to_dict(),
                            "ts_policy": self.ts_policy.params,
                        },
                        f,
                    )

    def eval(self):
        eval_metrics = dd(list)

        # run on eval batches
        # for _ in range(self.num_eval_batches):
        #     loss, metrics = jax.jit(self.loss_fn)(
        #         self.ts.params,
        #         self.ts_policy,
        #         next(self.eval_dataloader),
        #         next(self.rng_seq),
        #     )
        #     for k, v in metrics.items():
        #         eval_metrics[k].append(v)

        # for k, v in eval_metrics.items():
        #     eval_metrics[k] = jnp.mean(jnp.array(v))

        # run rollouts
        rollout_metrics = run_rollouts(
            rng=next(self.rng_seq),
            env=self.eval_envs,
            config=self.config,
            ts_policy=self.ts_policy,
            wandb_run=self.wandb_run,
        )
        eval_metrics.update(rollout_metrics)
        return eval_metrics
