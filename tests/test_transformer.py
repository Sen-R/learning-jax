from typing import List

import chex
import haiku as hk
import jax
import jax.numpy as jnp
import pytest

from learningjax.transformer import (
    FFN,
    MHA,
    MultiHeadLinear,
    TransformerLayer,
    _build_mask,
    _causal_mask,
    build_causal_transformer,
    compute_attention_matrix,
    multi_head_attention,
)


class TestFFN:
    def test_output_shape_and_intermediate_dimension(self) -> None:
        key = jax.random.PRNGKey(3243)
        seq = jnp.zeros((4, 8, 12, 16))
        net = hk.without_apply_rng(hk.transform(lambda x: FFN(name="test_ffn")(x)))
        params = net.init(key, seq)
        out = net.apply(params, seq)
        chex.assert_shape(out, (4, 8, 12, 16))
        chex.assert_shape(params["test_ffn/fc"]["w"], (16, 64))


class TestMultiHeadLinear:
    def test_shapes(self) -> None:
        key = jax.random.PRNGKey(23432)
        x = jnp.zeros((4, 8, 12, 16))
        proj = hk.without_apply_rng(hk.transform(lambda x: MultiHeadLinear(4, 3)(x)))
        params = proj.init(key, x)
        key = proj.apply(params, x)
        chex.assert_shape(key, (4, 8, 12, 4, 3))


class TestComputeAttentionMatrix:
    def test_computation(self) -> None:
        q_key, k_key = jax.random.split(jax.random.PRNGKey(82343), 2)
        mask = jnp.ones((13, 5)).at[(5, 2)].set(0.0)
        assert jnp.sum(mask).astype(jnp.int16) == 64  # Check mask set correctly
        q = jax.random.normal(q_key, shape=(2, 8, 13, 12, 3))
        k = jax.random.normal(k_key, shape=(2, 8, 5, 12, 3))
        logits = jnp.einsum("bdshf,bdShf->bdhsS", q, k) / jnp.sqrt(3)
        logits -= (1.0 - mask) * 1e30
        attn_matrix = jax.nn.softmax(logits)
        chex.assert_shape(attn_matrix, (2, 8, 12, 13, 5))
        chex.assert_trees_all_close(compute_attention_matrix(q, k, mask), attn_matrix)


class TestMultiHeadAttention:
    def test_output_shape(self) -> None:
        q_key, k_key, v_key = jax.random.split(jax.random.PRNGKey(98243), 3)
        q = jax.random.normal(q_key, shape=(2, 8, 5, 4, 3))
        k = jax.random.normal(k_key, shape=(2, 8, 6, 4, 3))
        v = jax.random.normal(v_key, shape=(2, 8, 6, 4, 7))
        out = multi_head_attention(q, k, v)
        chex.assert_shape(out, (2, 8, 5, 4, 7))


class TestMHA:
    def test_output_shape(self) -> None:
        p_key, a_key = jax.random.split(jax.random.PRNGKey(234), 2)
        layer = hk.transform(lambda x: MHA(4)(x))
        x = jnp.zeros((2, 8, 5, 16))
        params = layer.init(p_key, x)
        out = layer.apply(params, a_key, x)
        chex.assert_equal_shape((x, out))


class TestTransformerLayer:
    def test_output_shape(self) -> None:
        p_key, a_key = jax.random.split(jax.random.PRNGKey(234), 2)
        layer = hk.transform(lambda x: TransformerLayer(4)(x))
        x = jnp.zeros((2, 8, 5, 16))
        params = layer.init(p_key, x)
        out = layer.apply(params, a_key, x)
        chex.assert_equal_shape((x, out))


class TestCausalMask:
    @pytest.mark.parametrize(
        "nd,ns,expected",
        [(2, 2, [[1, 0], [1, 1]]), (2, 3, [[1, 1, 0], [1, 1, 1]])],
    )
    def test_outputs(self, nd: int, ns: int, expected: List[List[int]]) -> None:
        got = _causal_mask(nd, ns)
        chex.assert_trees_all_equal(got, jnp.array(expected))


class TestCausalTransformer:
    transformer: hk.Transformed
    params: hk.Params
    x: jax.Array

    @pytest.fixture(autouse=True)
    def _setup(self) -> None:
        vocab_size = 3
        context_size = 12
        embed_dim = 8
        num_layers = 1
        num_heads = 2
        self.transformer = build_causal_transformer(
            vocab_size, context_size, embed_dim, num_layers, num_heads
        )
        self.x = jnp.array([[0, 1, 2, 0, 1, 2], [0, 0, 1, 1, 2, 2]])
        self.params = self.transformer.init(jax.random.PRNGKey(42), self.x)
        self.mask = jnp.array([[1, 1, 1, 1, 1, 1], [1, 1, 1, 0, 0, 0]])  # arbitrary

    @pytest.mark.parametrize("use_mask", [True, False])
    def test_output_shape_without_mask(self, use_mask: bool) -> None:
        mask_in = self.mask if use_mask else None
        logits, mask_out = self.transformer.apply(
            self.params, jax.random.PRNGKey(42), self.x, mask_in
        )
        chex.assert_shape(logits, (2, 6, 3))
        if use_mask:
            chex.assert_trees_all_equal(mask_out, mask_in)
        else:
            assert mask_out is None

    def test_build_mask(self) -> None:
        x = jnp.array([[0, 1, 2], [0, 1, 2]])
        mask = jnp.array([[1, 1, 0], [1, 0, 0]])
        expected = jnp.array(
            [
                [[1, 0, 0], [1, 1, 0], [1, 1, 0]],
                [[1, 0, 0], [1, 0, 0], [1, 0, 0]],
            ]
        )
        chex.assert_trees_all_equal(_build_mask(x, mask), expected)
