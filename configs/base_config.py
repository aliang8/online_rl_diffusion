from ml_collections import ConfigDict, FieldReference, FrozenConfigDict, config_flags


def get_config():
    config = ConfigDict()
    config.seed = 0
    config.exp_name = "online_rl_diffusion"
    config.root_dir = "/scr/aliang80/online_rl_diffusion"
    config.video_dir = "videos"
    config.ckpt_dir = "model_ckpts"
    config.hidden_size = 128
    config.eval_every = 100
    config.save_every = 100
    config.num_training_episodes = 5000
    config.num_eval_episodes = 10
    config.results_dir = "results"
    config.mode = "train"
    config.save_eval_video = False
    config.num_eval_video_save = 2
    config.policy = "gaussian"  # vae

    # env
    config.env_id = "CartPole-v1"  # "MountainCarContinuous-v0"
    config.action_scale = 2.0

    # rl training
    config.policy_lr = 3e-4
    config.critic_lr = 1e-3
    config.gamma = 0.99
    config.entropy_weight = 0.01
    config.value_coeff = 0.5

    # value function
    config.baseline = False
    config.share_backbone = False

    # vae policy
    config.latent_dim = 16
    config.kl_div_weight = 1e-2

    # wandb
    config.disable_tqdm = False
    config.save_video = False
    config.smoke_test = True
    config.use_wb = False
    config.group_name = ""
    config.notes = ""
    config.tags = ()
    config.visualize = False
    return config
