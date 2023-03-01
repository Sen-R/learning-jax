from itertools import repeat
from typing import Tuple

import chex
import jax
import jax.numpy as jnp
import pytest

import learningjax.linear_regression as lr
from learningjax import utils

params = (
    (3.0, 2.0),
    (jnp.array([1.0, 2.0]), 3.0),
    (jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]), jnp.array([7.0, 8.0])),
)

x_y: Tuple[Tuple[chex.Array, chex.Array], ...] = (
    (0.5, 3.5),
    (jnp.array([0.5, 1.0]), 5.5),
    (jnp.array([0.5, 1.0, 2.0]), jnp.array([20.5, 25.0])),
)

x_y_shape = tuple((jnp.shape(x), jnp.shape(y)) for x, y in x_y)


class TestLinearModelParameters:
    @pytest.mark.parametrize("w,b", params)
    def test_init(self, w: chex.Array, b: chex.Array) -> None:
        params = lr.LinearModelParameters(w=w, b=b)
        chex.assert_trees_all_equal(params.w, w)
        chex.assert_trees_all_equal(params.b, b)

    def test_incompatible_params_raise(self) -> None:
        with pytest.raises(ValueError):
            lr.LinearModelParameters(w=jnp.array([[1.0, 2.0]]), b=3.0)

    @pytest.mark.parametrize(
        "f_s,o_s,w_s,b_s",
        (
            ((), (), (), ()),
            ((3,), (), (3,), ()),
            ((3,), (2,), (3, 2), (2,)),
        ),
    )
    def test_initialize(
        self,
        f_s: Tuple[int, ...],
        o_s: Tuple[int, ...],
        w_s: Tuple[int, ...],
        b_s: Tuple[int, ...],
    ) -> None:
        key_it = repeat(jax.random.PRNGKey(2343))
        params = lr.LinearModelParameters.initialize(key_it, f_s, o_s)
        assert jnp.shape(params.w) == w_s
        assert jnp.shape(params.b) == b_s

    @pytest.mark.parametrize(
        "f_s,o_s",
        (
            ((3, 2), ()),
            ((3,), (4, 5)),
            ((), (4,)),
        ),
    )
    def test_bad_initialize_inputs(
        self, f_s: Tuple[int, ...], o_s: Tuple[int, ...]
    ) -> None:
        key_it = repeat(jax.random.PRNGKey(2343))
        with pytest.raises(ValueError):
            lr.LinearModelParameters.initialize(key_it, f_s, o_s)


class TestLinearModel:
    @pytest.mark.parametrize("w,b,x,y", [a + b for a, b in zip(params, x_y)])
    def test_forward(
        self, w: chex.Array, b: chex.Array, x: chex.Array, y: chex.Array
    ) -> None:
        params = lr.LinearModelParameters(w=w, b=b)
        y_actual = lr.linear_model(params, x)
        chex.assert_trees_all_close(y_actual, y)


class TestCreateDataset:
    @pytest.mark.parametrize("w,b,x_s,y_s", [a + b for a, b in zip(params, x_y_shape)])
    def test_output_shapes_as_expected(
        self, w: chex.Array, b: chex.Array, x_s: Tuple[int, ...], y_s: Tuple[int, ...]
    ) -> None:
        n_samples = 10
        key_it = utils.key_generator(seed=80234)
        params = lr.LinearModelParameters(w=w, b=b)
        X, y = lr.create_dataset(key_it, n_samples, params, noise_scale=0.0)
        chex.assert_shape(X, (n_samples, *x_s))
        chex.assert_shape(y, (n_samples, *y_s))

    def test_noise_scaling(self) -> None:
        n_samples = 5
        params = lr.LinearModelParameters(w=1.0, b=-1.0)
        key_it = repeat(jax.random.PRNGKey(89243))

        datasets = {
            noise: lr.create_dataset(key_it, n_samples, params, noise)
            for noise in (0.0, 0.1, 1.0)
        }

        # Check X for each dataset is the same
        chex.assert_trees_all_close(datasets[0.0][0], datasets[0.1][0])
        chex.assert_trees_all_close(datasets[0.0][0], datasets[0.1][0])

        residuals = {noise: datasets[noise][1] - datasets[0.0][1] for noise in datasets}

        # Check that noise has actually been added
        assert jnp.max(jnp.abs(residuals[0.1])) > 0.05  # meaningfully nonzero

        # Check that noise for 1.0 dataset is 10x noise for 0.1 dataset
        chex.assert_trees_all_close(residuals[1.0], residuals[0.1] * 10.0)
