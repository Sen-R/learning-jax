from typing import Iterator, Tuple

import chex
import jax
import jax.numpy as jnp


@chex.dataclass
class LinearModelParameters:
    w: chex.Array
    b: chex.Array

    def __post_init__(self) -> None:
        w_shape = jnp.shape(self.w)
        b_shape = jnp.shape(self.b)
        if w_shape[1:] != b_shape:
            raise ValueError(
                f"Incompatible weight and bias parameters, got shapes: {w_shape} "
                f"and {b_shape}"
            )

    @classmethod
    def initialize(
        cls,
        key_it: Iterator[chex.PRNGKey],
        features_shape: Tuple[int, ...],
        outputs_shape: Tuple[int, ...],
    ) -> "LinearModelParameters":
        if len(features_shape) > 1:
            raise ValueError(
                f"Features must be at most 1 dimensional, got: {features_shape}"
            )
        if len(outputs_shape) > 1:
            raise ValueError(
                f"Outputs must be at most 1 dimensional, got: {outputs_shape}"
            )
        if len(outputs_shape) > 0 and len(features_shape) == 0:
            raise ValueError(
                "When multiple outputs, cannot have scalar feature, "
                f"got: {features_shape}"
            )
        w_shape = features_shape + outputs_shape
        b_shape = outputs_shape
        w = jax.random.normal(next(key_it), shape=w_shape)
        b = jax.random.normal(next(key_it), shape=b_shape)
        return cls(w=w, b=b)


def linear_model(params: LinearModelParameters, features: chex.Array) -> chex.Array:
    return jnp.dot(features, params.w) + params.b  # type: ignore


def create_dataset(
    key_it: Iterator[chex.PRNGKey],
    n_samples: int,
    params: LinearModelParameters,
    noise_scale: float,
) -> Tuple[chex.Array, chex.Array]:
    features_shape = jnp.shape(params.w)[:1]
    outputs_shape = jnp.shape(params.b)
    X = jax.random.normal(next(key_it), shape=(n_samples, *features_shape))
    y = jax.vmap(linear_model, in_axes=(None, 0))(params, X)
    chex.assert_shape(y, (n_samples, *outputs_shape))
    noise = noise_scale * jax.random.normal(next(key_it), shape=y.shape)
    return X, y + noise
