from typing import Optional

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
    """Module that simultaneously performs a linear projection to multiple spaces."""

    def __init__(
        self,
        num_heads: int,
        output_size: int,
        w_init: Optional[hk.initializers.Initializer] = None,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(name=name)
        self.num_heads = num_heads
        self.output_size = output_size
        self.w_init = w_init or hk.initializers.VarianceScaling()

    def __call__(self, x: jax.Array) -> jax.Array:
        *leading_dims, _ = x.shape
        layer = hk.Linear(self.num_heads * self.output_size, w_init=self.w_init)
        return layer(x).reshape((*leading_dims, self.num_heads, self.output_size))


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
        query = MultiHeadLinear(self.num_heads, key_size, self.w_init, "query")(x)
        key = MultiHeadLinear(self.num_heads, key_size, self.w_init, "key")(x)
        value = MultiHeadLinear(self.num_heads, key_size, self.w_init, "value")(x)

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
        self.ln_0 = hk.LayerNorm(-1, True, True)
        self.ln_1 = hk.LayerNorm(-1, True, True)
        self.mha = MHA(num_heads, w_init, name="mha")
        self.ffn = FFN(d_ff_multiplier, w_init, name="ffn")
        self.dropout_rate = dropout_rate

    def __call__(self, x: jax.Array, mask: Optional[jax.Array] = None) -> jax.Array:
        mha_out = self.mha(self.ln_0(x), mask)
        x = x + hk.dropout(hk.next_rng_key(), self.dropout_rate, mha_out)

        ffn_out = self.ffn(self.ln_1(x))
        x = x + hk.dropout(hk.next_rng_key(), self.dropout_rate, ffn_out)

        return x
