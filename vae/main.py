from trainer import Trainer
from absl import logging
from ml_collections import ConfigDict, FieldReference, FrozenConfigDict
import jax.numpy as jnp
import sys
import jax
import changepoint_aug.online_rl_diffusion.viz_helper as viz_helper
import os
import gymnasium as gym
import tqdm
import time
import collections
from collections import deque
import numpy as np
import gymnax
from gymnax.visualize import Visualizer


os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.01"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


def get_config():
    config = ConfigDict()

    config.project_name = "online_rl_diffusion"
    config.restore = ""
    config.dry_run = True
    config.log_level = logging.INFO

    seed = 42
    grad_acc = FieldReference(1)
    hidden_size = FieldReference(128)
    latent_size = FieldReference(4)

    config.training_steps = FieldReference(2000)
    config.seed = seed
    config.ckpt_dir = "/scr/aliang80/changepoint_aug/online_rl_diffusion/results"
    config.log_interval = 50
    config.ckpt_interval = 50
    config.eval_interval = 100
    config.visualize_every = 100
    config.batch_size = 8
    config.entropy_weight = 0.2
    config.entropy_decay_rate = 0.9995

    config.experiment_kwargs = ConfigDict(
        dict(
            dry_run=config.dry_run,
            seed=seed,
            batch_size=config.batch_size,
            ckpt_dir=config.ckpt_dir,
            discount=1.0,
            lr=2e-4,
            vf_lr=2e-5,
            kl_weight=1e-2,
            reward_weight=1.0,
            prior_loss_weight=1.0,
            prior_type="state_conditioned",
            # env_id="Pendulum-v1",
            env_id="CartPole-v1",
            # max_episode_steps=200,
            num_eval_episodes=5,
            policy_cls="categorical",  # vae, categorical
            max_episode_steps=1000,
            num_epochs=config.training_steps,
            use_lr_scheduler=False,
            train=dict(
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
                value_function=dict(hidden_size=hidden_size),
                encoder=dict(hidden_size=hidden_size, latent_size=latent_size),
                decoder=dict(
                    hidden_size=hidden_size, action_dim=1, is_discrete=False
                ),  # these get changed in the code
                standard=dict(latent_size=latent_size),
                flow=dict(
                    latent_size=latent_size, num_flows=2, hidden_size=hidden_size
                ),
                state_conditioned=dict(
                    hidden_size=hidden_size, latent_size=latent_size
                ),
                policy=dict(hidden_size=hidden_size, action_dim=1, is_discrete=False),
            ),
        )
    )

    config.lock()

    return config


if __name__ == "__main__":
    """
    Implement REINFORCE algorithm with VAE decoder as the policy.

    Includes variance reductiong with baseline subtraction.
    Estimates a value function based on the rollout returns.

    goal: maximize the expected return
    J(θ) = E[∑t r(s_t, a_t) - b(s_t)] = E[∑t r(s_t, a_t) - V(s_t)]
    grad J(θ) = E[∑t grad log π(a_t|s_t) * (r(s_t, a_t) - V(s_t))]

    where V(s_t) is the value function, which we estimate with a neural network

    we estimate log π(a_t|s_t) with the ELBO of the VAE.
    ^^ not sure if this is valid
    """
    config = get_config()
    config = FrozenConfigDict(config)
    print(config.experiment_kwargs)
    trainer = Trainer(config.experiment_kwargs)
    train_state = trainer.init_state

    # setup logging
    logging.set_verbosity(config.log_level)

    video_imgs = []

    rewards, gm_loss, policy_loss = [], [], []

    entropy_weight = config.entropy_weight

    scores_deque = deque(maxlen=100)
    jit_rollout = jax.jit(trainer.collect_rollout)
    jit_update_step = jax.jit(trainer._update_step)

    gamma = config.experiment_kwargs.discount

    for step in tqdm.tqdm(range(config.training_steps)):
        rng_key = next(trainer.rng_seq)
        start = time.time()
        obss, actions, returns, next_obs, dones_mask = jit_rollout(train_state, rng_key)
        end = time.time()
        # print("rollout time: ", (end - start))

        scores_deque.append(returns[0].item())
        rng_key = next(trainer.rng_seq)
        start_time = time.time()
        train_state, metrics = jit_update_step(
            train_state,
            rng_key,
            obss,
            actions,
            rewards,
            dones_mask,
            returns,
            entropy_weight,
        )
        lr = trainer.lr_fn(train_state.step)
        end_time = time.time()
        # print(f"step time: {(end_time - start_time)}")
        # print("done updating step")
        trainer.train_state = train_state

        # print("params")
        # print(state.params["state_conditioned_prior/linear"]["w"].mean())
        # print(state.params["decoder/linear"]["w"].mean())
        # print(state.params["encoder/linear"]["w"].mean())
        # print(train_state.params[0]["policy/linear"]["w"].mean())

        entropy_weight *= config.entropy_decay_rate

        if step % config.log_interval == 0:
            # elbo = metrics["elbo"]
            # reward = metrics["reward"]
            # pg_loss = metrics["pg_loss"]
            # recon_loss = metrics["recon_loss"]

            # logging.info(
            #     f"Step {step}, reward: {reward}, return: {rewards.sum()}, ELBO: {elbo}, PG loss: {pg_loss}, Recon loss: {recon_loss}, entropy weight: {entropy_weight}"
            # )
            import pprint

            pprint.pprint(metrics)
            print("lr: ", lr)
        if step % config.ckpt_interval == 0:
            trainer.save_checkpoint(train_state, step, config.ckpt_dir)

        if step % config.eval_interval == 0:
            print("average train reward: ", np.mean(scores_deque))
        # trainer.run_eval(train_state)
        #     rewards.append(metrics["reward"])
        #     gm_loss.append(metrics["elbo"])
        #     policy_loss.append(metrics["pg_loss"])

        #     actions = input
        #     # actions = trainer._sample(state, 1000)
        #     # actions = jax.random.normal(state.rng_key, (1000, config.action_dim))
        #     print(actions.min(), actions.max())
        #     viz_helper.clear_output(wait=True)
        #     plot = viz_helper.make_plot(
        #         actions,
        #         trainer.reward,
        #         rewards,
        #         gm_loss,
        #         policy_loss,
        #         visualize_every=config.visualize_every,
        #         total_steps=config.training_steps,
        #         label="VAE",
        #     )
        #     video_imgs.append(plot)

    # save plots to a gif
    # viz_helper.animate(video_imgs, filename="vae_bimodal_2.gif", fps=5)
