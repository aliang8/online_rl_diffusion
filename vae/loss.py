import jax
import optax
import jax.numpy as jnp
import haiku as hk


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
    entropy_weight,
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
    policy_params, value_fn_params = params
    value_estimate = (
        hk.without_apply_rng(value_fn).apply(value_fn_params, obs).squeeze(-1)
    )

    # calculate policy gradient loss
    action, logits, action_log_prob = policy.apply(policy_params, rng_key, obs)

    # PG = E[log(pi(a|s)) * r(s)]
    pg_loss = -action_log_prob * (returns - value_estimate)
    pg_loss = pg_loss.sum() / dones_mask.sum()

    value_estimate_loss = optax.squared_error(returns - value_estimate)
    value_estimate_loss = value_estimate_loss.sum() / dones_mask.sum()

    total_loss = pg_loss + value_estimate_loss
    return total_loss, {"pg_loss": pg_loss, "value_estimate_loss": value_estimate_loss}


def vae_policy_loss_fn(
    params, rng_key, obs, actions, rewards, dones_mask, returns, entropy_weight
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
    value_estimate = value_fn.apply(value_fn_params, None, obs)

    # compute return
    # Recalculate the total reward applying discounted factor

    # run forward pass
    _, mean, var, recon, z, kl, kl_to_learned_prior = model.apply(
        vae_params, rng_key, obs, actions
    )

    if self.is_discrete:
        # compute cross entropy
        recon_loss = optax.softmax_cross_entropy_with_integer_labels(
            recon, actions.squeeze(-1)
        )
    else:
        # compute MSE for reconstruction loss, this is log-likelihood
        recon_loss = jnp.square(actions - recon).mean(axis=-1)

    # compute KL divergence between unit Gaussian and Gaussian(mean, var)
    # scale is std
    # p = dist.Normal(loc=jnp.zeros_like(mean), scale=jnp.ones_like(var))
    # q = dist.Normal(loc=mean, scale=jnp.sqrt(var))
    # kl_to_uniform_prior = dist.kl_divergence(q, p).sum(axis=-1)

    # ELBO is -likelihood - KL, want to maximize this term
    elbo = -recon_loss - config.kl_weight * kl

    # apply dones mask
    elbo = elbo * dones_mask

    # jax.debug.breakpoint()
    # elbo_loss = recon_loss + config.kl_weight * kl_to_uniform_prior

    # add entropy to encourage exploration and not get stuck at some local minima
    # elbo is proxy for log(p(a|s))
    # we want to explore actions that have low log(p(a|s)), we penalize if elbo is high
    # r = self.reward(input).squeeze(-1)
    r = rewards.mean()
    # r_ent = rewards * config.reward_weight - entropy_weight * elbo
    # r_ent = (
    #     self.reward(input).squeeze(-1) * config.reward_weight
    #     - entropy_weight * elbo
    # )
    pg_loss = -((returns - value_estimate) * elbo)
    # masked mean
    pg_loss = pg_loss.sum() / dones_mask.sum()

    value_estimate_loss = optax.squared_error(returns - value_estimate)
    # apply dones mask
    value_estimate_loss = value_estimate_loss * dones_mask
    value_estimate_loss = value_estimate_loss.sum() / dones_mask.sum()

    kl_to_learned_prior = kl_to_learned_prior * dones_mask
    kl_to_learned_prior = kl_to_learned_prior.sum() / dones_mask.sum()

    total_loss = pg_loss + 0.01 * kl_to_learned_prior + value_estimate_loss

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
