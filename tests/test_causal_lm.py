from typing import List

import chex
import jax
import jax.numpy as jnp
import numpy as np

from learningjax.causal_lm import build_loss_fn, build_metric_fn, pad_and_convert_batch


def test_pad_and_convert_batch() -> None:
    input_lists: List[List[int]] = [[0, 1, 2], [4, 5, 6, 7], [], [8, 9]]
    batch = {"input_ids": input_lists}
    got = pad_and_convert_batch(batch, 3, 99)
    expected = np.array([[0, 1, 2], [4, 5, 6], [99, 99, 99], [8, 9, 99]])
    chex.assert_trees_all_equal(got, expected)


# Set up dummy inputs for testing loss and metric functions
vocab_size = 6
pad_token_id = vocab_size - 1
batch = jnp.array([[0, 1, 2, 4], [2, 1, pad_token_id, pad_token_id]])
dummy_model_output = {"logits": jnp.zeros((2, 3, vocab_size))}


def test_build_metric_fn() -> None:
    compute_metric = build_metric_fn(
        lambda _, __, ___: dummy_model_output,
        pad_token_id=pad_token_id,
    )
    got = jax.jit(compute_metric)({}, None, batch)
    expected = jnp.log(vocab_size) * jnp.array([2, 2, 1]), jnp.array([2, 2, 1])
    chex.assert_trees_all_close(got, expected)


def test_build_loss_fn() -> None:
    compute_loss = build_loss_fn(
        lambda _, __, ___: dummy_model_output, pad_token_id=pad_token_id
    )
    got = jax.jit(compute_loss)({}, None, batch)
    expected = jnp.log(vocab_size)
    chex.assert_tree_all_close(got, expected)
