from typing import Tuple

import jax
import jax.numpy as jnp
import optax  # type: ignore


def mse(y_true: jax.Array, y_pred: jax.Array) -> jax.Array:
    return jnp.mean(jnp.subtract(y_true, y_pred) ** 2)


def weighted_softmax_cross_entropy_with_integer_labels(
    logits: jax.Array, labels: jax.Array, sample_weights: jax.Array
) -> Tuple[jax.Array, jax.Array]:
    x_ents: jax.Array = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
    weighted_x_ents = x_ents * sample_weights
    total_weighted_x_ents = jnp.sum(weighted_x_ents)
    total_weight = jnp.sum(sample_weights)
    mean_loss = total_weighted_x_ents / total_weight
    return mean_loss, total_weight
