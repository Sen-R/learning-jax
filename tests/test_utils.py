import jax
import jax.numpy as jnp

from learningjax.utils import key_generator


def test_key_generator() -> None:
    key_iterator = key_generator(seed=42)
    first_key = next(key_iterator)
    assert isinstance(first_key, jax.Array)
    next_key = next(key_iterator)
    assert not jnp.array_equal(first_key, next_key)
