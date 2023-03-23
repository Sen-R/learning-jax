from typing import Optional

import chex
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np


class FFN(hk.Module):
    """Feed forward sublayer for a transformer network."""

    def __init__(
        self,
        d_ff_multiplier: int = 4,
        w_init: Optional[hk.initializers.Initializer] = None,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(name=name)
        self.d_ff_multiplier = d_ff_multiplier
        self.w_init = w_init or hk.initializers.VarianceScaling()

    def __call__(self, x: jax.Array) -> jax.Array:
        *leading_dims, seq_len, d_model = x.shape
        d_ff = d_model * self.d_ff_multiplier
        h = jax.nn.gelu(hk.Linear(d_ff, name="fc")(x))
        h2 = hk.Linear(d_model, name="out")(h)
        return h2


class MultiHeadLinear(hk.Module):
    """Module that simultaneously performs multiple affine transformations."""

    def __init__(
        self,
        num_heads: int,
        output_size: int,
        with_bias: bool = True,
        w_init: Optional[hk.initializers.Initializer] = None,
        b_init: Optional[hk.initializers.Initializer] = None,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(name=name)
        self.num_heads = num_heads
        self.output_size = output_size
        self.with_bias = with_bias
        self.w_init = w_init or hk.initializers.VarianceScaling()
        self.b_init = b_init or hk.initializers.Constant(0.0)

    def __call__(self, x: jax.Array) -> jax.Array:
        *leading_dims, d_f = x.shape
        d_full = self.num_heads * self.output_size

        w = hk.get_parameter("w", shape=(d_f, d_full), dtype=x.dtype, init=self.w_init)
        flat_out = jnp.dot(x, w)

        if self.with_bias:
            b = hk.get_parameter("b", shape=(d_full,), dtype=x.dtype, init=self.b_init)
            flat_out = flat_out + b

        return flat_out.reshape((*leading_dims, self.num_heads, self.output_size))


def compute_attention_matrix(
    query: jax.Array, key: jax.Array, mask: Optional[jax.Array] = None
) -> jax.Array:
    sqrt_d_k = np.sqrt(key.shape[-1])
    attention_logits = jnp.einsum("...shf,...Shf->...hsS", query, key) / sqrt_d_k
    if mask is not None:
        attention_logits = jnp.where(mask, attention_logits, -1e30)
    return jax.nn.softmax(attention_logits)


def multi_head_attention(
    query: jax.Array, key: jax.Array, value: jax.Array, mask: Optional[jax.Array] = None
) -> jax.Array:
    attention_matrix = compute_attention_matrix(query, key, mask)
    return jnp.einsum("...hsS,...Shf->...shf", attention_matrix, value)


class MHA(hk.Module):
    """Multi-head self-attention sublayer for a transformer network."""

    def __init__(
        self,
        num_heads: int,
        w_init: Optional[hk.initializers.Initializer] = None,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(name=name)
        self.num_heads = num_heads
        self.w_init = w_init or hk.initializers.VarianceScaling()

    def __call__(self, x: jax.Array, mask: Optional[jax.Array] = None) -> jax.Array:
        *leading_dims, seq_len, d_model = x.shape

        # Create query, key and value sequences
        key_size = d_model // self.num_heads
        query, key, value = [
            MultiHeadLinear(self.num_heads, key_size, w_init=self.w_init, name=name)(x)
            for name in ("query", "key", "value")
        ]

        # Compute attention and project combined outputs back to original vector space
        attended_value = multi_head_attention(query, key, value)
        concatenated = attended_value.reshape(
            (*leading_dims, seq_len, self.num_heads * key_size)
        )
        projector = hk.Linear(d_model, w_init=self.w_init, name="out")
        output = projector(concatenated)
        return output


class TransformerLayer(hk.Module):
    """Single transformer layer with pre-layer norm."""

    def __init__(
        self,
        num_heads: int,
        d_ff_multiplier: int = 4,
        dropout_rate: float = 0.0,
        w_init: Optional[hk.initializers.Initializer] = None,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(name=name)
        self.num_heads = num_heads
        self.d_ff_multiplier = d_ff_multiplier
        self.dropout_rate = dropout_rate
        self.w_init = w_init

    def __call__(self, x: jax.Array, mask: Optional[jax.Array] = None) -> jax.Array:
        ln_0 = hk.LayerNorm(-1, True, True, name="ln_pre_mha")
        ln_1 = hk.LayerNorm(-1, True, True, name="ln_pre_ffn")
        mha = MHA(self.num_heads, self.w_init, name="mha")
        ffn = FFN(self.d_ff_multiplier, self.w_init, name="ffn")

        mha_out = mha(ln_0(x), mask)
        x = x + hk.dropout(hk.next_rng_key(), self.dropout_rate, mha_out)

        ffn_out = ffn(ln_1(x))
        x = x + hk.dropout(hk.next_rng_key(), self.dropout_rate, ffn_out)

        return x


def build_transformer(
    vocab_size: int, context_size: int, embed_dim: int, num_layers: int, num_heads: int
) -> hk.Transformed:
    @hk.transform
    def forward(x: jax.Array, mask: Optional[jax.Array] = None) -> jax.Array:
        embed_init = hk.initializers.VarianceScaling(mode="fan_out")
        with hk.experimental.name_scope("token_embedding"):
            wte = hk.get_parameter("w", shape=(vocab_size, embed_dim), init=embed_init)
        with hk.experimental.name_scope("position_embedding"):
            wpe = hk.get_parameter(
                "w", shape=(context_size, embed_dim), init=embed_init
            )
        ln_final = hk.LayerNorm(-1, True, True, name="ln_final")
        layers = [
            TransformerLayer(num_heads, name=f"layer_{i}") for i in range(num_layers)
        ]
        chex.assert_type(x, jnp.int32)
        assert x.shape[-1] <= context_size
        token_embeddings = wte[(x,)]
        position_embeddings = wpe[: x.shape[-1]]
        z = token_embeddings + position_embeddings
        for layer in layers:
            z = layer(z, mask)
        z = ln_final(z)
        logits = jnp.dot(z, wte.T)
        return logits

    return forward
