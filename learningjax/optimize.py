from typing import TypeVar

import chex
import jax

Params = TypeVar("Params")


def sgd(grads: chex.ArrayTree, params: Params, alpha: float) -> Params:
    return jax.tree_map(lambda g, p: p - alpha * g, grads, params)  # type: ignore
