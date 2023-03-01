import chex
import jax.numpy as jnp
import pytest

from learningjax import loss


@pytest.mark.parametrize(
    "y_t,y_p,mse",
    ((1.0, 0.5, 0.25), (jnp.array([1.0, 2.0]), jnp.array([0.9, 2.2]), 0.025)),
)
def test_mse(y_t: chex.Array, y_p: chex.Array, mse: float) -> None:
    mse_actual = loss.mse(y_t, y_p)
    chex.assert_trees_all_close(mse_actual, mse)
