import chex
import jax.numpy as jnp
import pytest

import learningjax.optimize as opt


@pytest.mark.parametrize(
    "grads,params,new_params",
    (
        (jnp.array([0.5, 1.0]), jnp.array([1.0, 2.0]), jnp.array([0.95, 1.9])),
        ((0.5, 1.0), (1.0, 2.0), (0.95, 1.9)),
        (
            {"w": jnp.array([0.2, 0.5]), "b": 1.0},
            {"w": jnp.array([1.0, 2.0]), "b": 3.0},
            {"w": jnp.array([0.98, 1.95]), "b": 2.9},
        ),
    ),
)
def test_sgd(
    grads: chex.ArrayTree, params: chex.ArrayTree, new_params: chex.ArrayTree
) -> None:
    new_params_actual = opt.sgd(grads, params, alpha=0.1)
    chex.assert_trees_all_close(new_params_actual, new_params)
