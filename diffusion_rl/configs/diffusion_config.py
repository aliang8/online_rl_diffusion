from diffusion_rl.configs.base_config import get_config as get_base_config
from ml_collections.config_dict import ConfigDict


def get_config(config_string: str = None):
    config = get_base_config(config_string)
    config.trainer = "offline"
    config.env_id = "halfcheetah-medium-v2"

    config.lr = 3e-4
    config.eps = 1e-8
    config.num_epochs = 1000
    config.num_iter_per_epoch = 100
    config.batch_size = 256

    # for rendering images
    config.width = 100
    config.height = 100

    policy_config = ConfigDict()
    policy_config.name = "diffusion"
    # diffusion policy hyperparameters (DDPM)
    policy_config.beta_schedule = "linear"
    policy_config.n_timesteps = 5
    policy_config.loss_type = "l2"
    policy_config.predict_epsilon = True
    policy_config.hidden_size = 256
    policy_config.time_embed_dim = 16
    policy_config.conditional = True

    config.policy = policy_config

    config.keys_to_include = {
        "env_id": None,
        "num_epochs": None,
        "num_iter_per_epoch": None,
        "policy": [
            "name",
            "n_timesteps",
            "beta_schedule",
            "time_embed_dim",
            "conditional",
        ],
    }
    return config
