import jax
import optax
import jax.numpy as jnp
import haiku as hk
import tree
from tensorflow_probability.substrates import jax as tfp

dist = tfp.distributions
eps = jnp.finfo(jnp.float32).eps.item()


def policy_loss_fn(
    params,
    rng_key,
    value_fn,
    policy,
    obs,
    actions,
    rewards,
    dones_mask,
    returns,
):
    """
    ELBO loss: E_p[log(x)] - KL(d||q), where p ~ Be(0.5) and q ~ N(0,1).

    PG loss = -(r(s) * log(pi(a|s)) - H(pi(a|s))
    """
    # input shape is (B, 1)
    obs = jax.lax.stop_gradient(obs)

    # compute value estimate V(s)
    policy_params, value_fn_params = params
    value_estimate = (
        hk.without_apply_rng(value_fn).apply(value_fn_params, obs).squeeze(-1)
    )

    _, logits, _ = policy.apply(policy_params, rng_key, obs)

    # compute action log probabilities
    dist = tfp.distributions.Categorical(logits=logits)
    a_log_probs = dist.log_prob(actions)

    # PG = E[log(pi(a|s)) * r(s)]
    returns = (returns - jnp.mean(returns)) / (jnp.std(returns) + eps)
    # pg_loss = -a_log_probs * (returns - value_estimate)
    pg_loss = -a_log_probs * returns
    pg_loss *= dones_mask
    pg_loss = pg_loss.sum()

    value_estimate_loss = optax.squared_error(returns - value_estimate)
    value_estimate_loss *= dones_mask
    value_estimate_loss = value_estimate_loss.sum() / dones_mask.sum()

    # jax.debug.breakpoint()
    # total_loss = pg_loss + value_estimate_loss
    total_loss = pg_loss

    # apply L2 regularization on the weights
    l2_reg = sum((1e-3 * optax.l2_loss(w)).mean() for w in jax.tree_leaves(params))
    # total_loss += l2_reg
    return total_loss, {
        "pg_loss": pg_loss,
        "value_estimate_loss": value_estimate_loss,
        "l2_reg": l2_reg,
    }


def vae_policy_loss_fn(
    params,
    rng_key,
    value_fn,
    policy,
    obs,
    actions,
    rewards,
    dones_mask,
    returns,
    entropy_weight,
    kl_weight,
    prior_loss_weight,
    is_discrete,
):
    """
    ELBO loss: E_p[log(x)] - KL(d||q), where p ~ Be(0.5) and q ~ N(0,1).

    PG loss = -(r(s) * log(pi(a|s)) - H(pi(a|s))
    """
    # input shape is (B, 1)
    obs = jax.lax.stop_gradient(obs)
    actions = jax.lax.stop_gradient(actions)
    if len(actions.shape) == 1:
        actions = jnp.expand_dims(actions, axis=1)

    # compute value estimate V(s)
    vae_params, value_fn_params = params
    value_estimate = (
        hk.without_apply_rng(value_fn).apply(value_fn_params, obs).squeeze(-1)
    )

    # run forward pass
    _, mean, var, recon, z, kl, kl_to_learned_prior = policy.apply(
        vae_params, rng_key, obs, actions
    )

    if is_discrete:
        # compute cross entropy
        recon_loss = optax.softmax_cross_entropy_with_integer_labels(
            recon, actions.squeeze(-1)
        )
    else:
        # compute MSE for reconstruction loss, this is log-likelihood
        recon_loss = jnp.square(actions - recon).mean(axis=-1)

    # ELBO is -likelihood - KL, want to maximize this term
    # apply mask to zero out loss for done states
    elbo = -recon_loss - kl_weight * kl
    pg_loss = -((returns - value_estimate) * elbo)
    pg_loss *= dones_mask
    pg_loss = pg_loss.sum() / dones_mask.sum()

    value_estimate_loss = optax.squared_error(returns - value_estimate)
    value_estimate_loss *= dones_mask
    value_estimate_loss = value_estimate_loss.sum() / dones_mask.sum()

    kl_to_learned_prior *= dones_mask
    kl_to_learned_prior = kl_to_learned_prior.sum() / dones_mask.sum()

    total_loss = pg_loss + prior_loss_weight * kl_to_learned_prior + value_estimate_loss

    metrics = {
        "recon_loss": recon_loss.sum() / dones_mask.sum(),
        "kl": kl.sum() / dones_mask.sum(),
        "kl_to_learned_prior": kl_to_learned_prior,
        "elbo": elbo.sum() / dones_mask.sum(),
        "pg_loss": pg_loss,
        "reward": rewards.sum() / dones_mask.sum(),
        "value_estimate_loss": value_estimate_loss,
    }
    return total_loss, metrics
