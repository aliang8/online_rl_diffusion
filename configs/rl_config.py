from configs.base_config import get_base_config
from ml_collections import ConfigDict, FieldReference, FrozenConfigDict, config_flags


def get_config():
    config = get_base_config()
    config.results_file = "reinforce.pkl"
    config.vizdom_name = "reinforce"
    return config