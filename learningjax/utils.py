from typing import Any, Iterator

import jax


def key_generator(seed: Any) -> Iterator[jax.random.PRNGKeyArray]:
    key = jax.random.PRNGKey(seed)
    while True:
        key, subkey = jax.random.split(key)
        yield key
