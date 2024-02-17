from trainer import Trainer
from absl import logging
from ml_collections import ConfigDict, FieldReference, FrozenConfigDict
import jax.numpy as jnp
import sys
import jax

sys.path.append("/scr/aliang80/changepoint_aug/online_rl_diffusion/")
import viz_helper as viz_helper
import os

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.01"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


def get_config():
    config = ConfigDict()

    config.project_name = "online_rl_diffusion"
    config.restore = ""
    config.dry_run = True
    config.log_level = logging.INFO

    seed = 42
    action_dim = 1
    grad_acc = FieldReference(1)

    config.training_steps = FieldReference(5000)
    config.seed = seed
    config.ckpt_dir = "/scr/aliang80/changepoint_aug/online_rl_diffusion/results"
    config.log_interval = 500
    config.ckpt_interval = 5000
    config.eval_interval = 1000
    config.visualize_every = 100
    config.action_dim = action_dim
    config.batch_size = 128
    config.entropy_weight = 0.2
    config.entropy_decay_rate = 0.9995

    config.experiment_kwargs = ConfigDict(
        dict(
            dry_run=config.dry_run,
            seed=seed,
            action_dim=action_dim,
            batch_size=config.batch_size,
            ckpt_dir=config.ckpt_dir,
            lr=1e-3,
            kl_weight=1e-3,
            reward_weight=0.1,
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
                encoder=dict(
                    hidden_size1=16,
                    hidden_size2=16,
                    latent_size=4,
                ),
                decoder=dict(hidden_size1=16, hidden_size2=16, action_dim=action_dim),
            ),
        )
    )

    config.lock()

    return config


if __name__ == "__main__":
    config = get_config()
    config = FrozenConfigDict(config)
    print(config.experiment_kwargs)
    trainer = Trainer(config.experiment_kwargs)
    state = trainer.init_state

    # setup logging
    logging.set_verbosity(config.log_level)

    video_imgs = []

    rewards, gm_loss, policy_loss = [], [], []

    entropy_weight = config.entropy_weight

    for step in range(config.training_steps):
        # create linspace
        input = jnp.linspace(-1, 1, 5000).reshape(-1, 1)
        # evaluate logp
        logp = trainer._compute_logp(state, input)
        # select the top 1000
        input = jnp.array(input[logp.argsort()[-1000:]])

        # input = trainer._sample(state, config.batch_size)
        # sample from gaussian
        # input = (
        #     jax.random.normal(state.rng_key, (config.batch_size, config.action_dim))
        #     * 0.3
        # )
        # input = jnp.zeros((config.batch_size, config.action_dim))
        state, metrics = trainer._update_step(state, input, entropy_weight)
        trainer.state = state
        entropy_weight *= config.entropy_decay_rate

        if step % config.log_interval == 0:
            elbo = metrics["elbo"]
            reward = metrics["reward"]
            pg_loss = metrics["pg_loss"]
            recon_loss = metrics["recon_loss"]

            logging.info(
                f"Step {step}, reward: {reward}, ELBO: {elbo}, PG loss: {pg_loss}, Recon loss: {recon_loss}, entropy weight: {entropy_weight}"
            )
        if step % config.ckpt_interval == 0:
            trainer.save_checkpoint(state, step, config.ckpt_dir)

        # generate visualization
        if step % config.visualize_every == 0:
            rewards.append(metrics["reward"])
            gm_loss.append(metrics["elbo"])
            policy_loss.append(metrics["pg_loss"])

            actions = input
            # actions = trainer._sample(state, 1000)
            # actions = jax.random.normal(state.rng_key, (1000, config.action_dim))
            print(actions.min(), actions.max())
            viz_helper.clear_output(wait=True)
            plot = viz_helper.make_plot(
                actions,
                trainer.reward,
                rewards,
                gm_loss,
                policy_loss,
                visualize_every=config.visualize_every,
                total_steps=config.training_steps,
                label="VAE",
            )
            video_imgs.append(plot)

    # save plots to a gif
    viz_helper.animate(video_imgs, filename="vae_bimodal_2.gif", fps=5)
