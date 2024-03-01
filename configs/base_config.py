from ml_collections import ConfigDict, FieldReference, FrozenConfigDict, config_flags


def get_base_config():
    config = ConfigDict()
    config.seed = 0
    config.ray_experiment_name = "online_rl_diffusion"
    config.root_dir = "/scr/aliang80/online_rl_diffusion"
    config.video_dir = "videos"
    config.batch_size = 128
    config.hidden_size = 128
    config.eval_every = 50
    config.save_every = 10
    config.num_training_episodes = 500
    config.num_eval_episodes = 10
    config.max_episode_steps = 1000
    config.results_dir = "results"
    config.mode = "train"
    config.vizdom_name = ""
    config.env_name = "MountainCarContinuous-v0"
    config.smoke_test = True
    config.save_eval_video = True
    config.num_eval_video_save = 2

    # rl training
    config.policy_lr = 3e-4
    config.critic_lr = 3e-4
    config.gamma = 0.99
    config.entropy_weight = 0.1
    return config
