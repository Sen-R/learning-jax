import chex
import jax


def sgd(grads: chex.ArrayTree, params: chex.ArrayTree, alpha: float) -> chex.ArrayTree:
    return jax.tree_map(lambda g, p: p - alpha * g, grads, params)  # type: ignore
