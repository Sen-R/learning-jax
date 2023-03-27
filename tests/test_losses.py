from typing import Tuple

import chex
import jax
import jax.numpy as jnp
import pytest

from learningjax import losses


@pytest.mark.parametrize(
    "y_t,y_p,mse",
    ((1.0, 0.5, 0.25), (jnp.array([1.0, 2.0]), jnp.array([0.9, 2.2]), 0.025)),
)
def test_mse(y_t: jax.Array, y_p: jax.Array, mse: float) -> None:
    mse_actual = losses.mse(y_t, y_p)
    chex.assert_trees_all_close(mse_actual, mse)


@pytest.mark.parametrize(
    "logits,labels,sample_weights,expected",
    [
        (
            jnp.array([[0.0, 0.0], [-1.0, 0.5], [1.5, 0.5]]),
            jnp.array([0, 1, 1]),
            jnp.array([1.0, 0.5, 0.25]),
            (0.6412396, 1.75),
        ),
    ],
)
def test_weighted_softmax_cross_entropy_with_integer_labels(
    logits: jax.Array,
    labels: jax.Array,
    sample_weights: jax.Array,
    expected: Tuple[float, float],
) -> None:
    got = losses.weighted_softmax_cross_entropy_with_integer_labels(
        logits, labels, sample_weights
    )
    chex.assert_tree_all_close(got, expected)
