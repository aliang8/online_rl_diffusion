from typing import Dict
import gym
import jax
import wandb
import tqdm
import time
import einops
import numpy as np
from flax.training.train_state import TrainState


def run_rollouts(
    config,
    ts_policy,
    rng: jax.random.PRNGKey,
    env: gym.Env,
    wandb_run: None,
) -> Dict[str, float]:
    stats = {"return": [], "length": []}

    videos = []
    max_action = float(env.action_space.high[0])
    start = time.time()

    for _ in tqdm.tqdm(range(config.num_eval_rollouts)):
        observation, done = env.reset(), False

        if len(observation.shape) == 1:
            observation = np.expand_dims(observation, axis=0)

        imgs = []

        if config.visualize_rollouts:
            imgs.append(
                env.render(mode="rgb_array", width=config.width, height=config.height)
            )

        t = 0
        episode_return = 0.0

        while not done and t < env.spec.max_episode_steps:
            rng, policy_rng = jax.random.split(rng)
            action = ts_policy.apply_fn(ts_policy.params, policy_rng, cond=observation)
            action = np.clip(action, -max_action, max_action)

            observation, reward, done, info = env.step(np.array(action).squeeze(axis=0))
            episode_return += reward

            if len(observation.shape) == 1:
                observation = np.expand_dims(observation, axis=0)

            if config.visualize_rollouts:
                imgs.append(
                    env.render(
                        mode="rgb_array", width=config.width, height=config.height
                    )
                )
            t += 1

        stats["return"].append(episode_return)
        stats["length"].append(t)

        if config.visualize_rollouts:
            videos.append(np.stack(imgs))

    normalized_scores = [env.get_normalized_score(s) for s in stats["return"]]
    avg_norm_score = env.get_normalized_score(np.mean(stats["return"])) * 100
    std_norm_score = np.std(normalized_scores)

    for k, v in stats.items():
        stats[k] = np.mean(v)

    stats["norm_return"] = avg_norm_score
    stats["std_norm_return"] = std_norm_score

    end = time.time()
    stats["time"] = end - start

    if wandb_run is not None and config.visualize_rollouts:
        # generate the images
        videos = np.array(videos)
        videos = einops.rearrange(videos, "n t h w c -> n t c h w")
        wandb_run.log({"eval_rollouts": wandb.Video(np.array(videos), fps=30)})

    return stats
