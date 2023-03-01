import chex
import jax.numpy as jnp


def mse(y_true: chex.Array, y_pred: chex.Array) -> chex.Array:
    return jnp.mean(jnp.subtract(y_true, y_pred) ** 2)
