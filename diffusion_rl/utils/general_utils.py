import collections
from typing import Any, Optional
from absl import logging
from flax.training import orbax_utils
import haiku as hk
import jax
import jax.numpy as jnp
import orbax
import numpy as np
import tensorflow_probability
import tensorflow as tf

tfp = tensorflow_probability.substrates.jax
tfd = tensorflow_probability.substrates.jax.distributions


def format_dict_keys(dictionary, format_fn):
    """Returns new dict with `format_fn` applied to keys in `dictionary`."""
    return collections.OrderedDict(
        [(format_fn(key), value) for key, value in dictionary.items()]
    )


def prefix_dict_keys(dictionary, prefix):
    """Add `prefix` to keys in `dictionary`."""
    return format_dict_keys(dictionary, lambda key: "%s%s" % (prefix, key))
