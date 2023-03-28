from typing import Callable, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax  # type: ignore
from datasets import Dataset  # type: ignore
from tqdm import tqdm  # type: ignore

Params = Dict[str, Dict[str, jax.Array]]  # standard Haiku params type
ApplyFn = Callable[
    [Params, Optional[jax.random.KeyArray], jax.Array], Dict[str, jax.Array]
]
LossFn = Callable[[Params, Optional[jax.random.KeyArray], jax.Array], jax.Array]
MetricFn = Callable[
    [Params, Optional[jax.random.KeyArray], jax.Array], Tuple[jax.Array, jax.Array]
]


def _unreduced_weighted_loss(
    params: Params,
    key: Optional[jax.random.KeyArray],
    batch: jax.Array,
    model: ApplyFn,
    pad_token_id: int,
) -> Tuple[jax.Array, jax.Array]:
    inputs, labels = batch["input_ids"], batch["labels"]
    sample_weights = inputs != pad_token_id
    logits = model(params, key, inputs)["logits"]
    xents = (
        optax.softmax_cross_entropy_with_integer_labels(logits, labels) * sample_weights
    )
    return xents, sample_weights


def build_metric_fn(model: ApplyFn, pad_token_id: int) -> MetricFn:
    def compute_metric(
        params: Params,
        key: Optional[jax.random.KeyArray],
        batch: jax.Array,
    ) -> Tuple[jax.Array, jax.Array]:
        xents, sample_weights = _unreduced_weighted_loss(
            params, key, batch, model, pad_token_id
        )
        reduction_axes = range(xents.ndim - 1)  # Don't reduce over sequence dimension
        total_xents = jnp.sum(xents, axis=reduction_axes)
        total_weight = jnp.sum(sample_weights, axis=reduction_axes)
        return total_xents, total_weight

    return compute_metric


def build_loss_fn(model: ApplyFn, pad_token_id: int) -> LossFn:
    def compute_loss(
        params: Params, key: Optional[jax.random.KeyArray], batch: jax.Array
    ) -> jax.Array:
        x_ents, sample_weights = _unreduced_weighted_loss(
            params, key, batch, model, pad_token_id
        )
        return jnp.sum(x_ents) / jnp.sum(sample_weights)

    return compute_loss


def pad_and_convert_batch(
    batch: Dict[str, List[List[int]]], max_length: int, pad_token_id: int
) -> Dict[str, jax.Array]:
    tot_length = max_length + 1
    full = pad_token_id * np.ones((len(batch["input_ids"]), tot_length), dtype=np.int32)
    for i, row in enumerate(batch["input_ids"]):
        full[i, : min(tot_length, len(row))] = row[:tot_length]
    full_on_device = jax.device_put(full)
    return {"input_ids": full_on_device[..., :-1], "labels": full_on_device[..., 1:]}


def evaluate(
    params: Params,
    model: ApplyFn,
    dataset: Dataset,
    batch_size: int,
    key: jax.random.KeyArray,
    max_length: int,
    pad_token_id: int,
    max_steps: Optional[int] = None,
) -> Tuple[jax.Array, jax.Array]:
    """Evaluate logits model on cross entropy loss."""
    total_loss = jnp.zeros((max_length,))
    total_weights = jnp.zeros((max_length,))
    compute_metric = jax.jit(build_metric_fn(model, pad_token_id))
    num_batches = len(dataset) // batch_size
    max_steps = num_batches if max_steps is None else max_steps
    p_bar = tqdm(
        dataset.iter(batch_size=batch_size, drop_last_batch=True),
        desc="Starting",
        total=max_steps,
    )
    for step_no, batch in enumerate(p_bar):
        if step_no == max_steps:
            break
        processed = pad_and_convert_batch(batch, max_length, pad_token_id)
        key, subkey = jax.random.split(key)
        this_loss, this_weights = compute_metric(params, subkey, processed)
        total_loss = total_loss + this_loss
        total_weights = total_weights + this_weights
        loss_so_far = jnp.sum(total_loss) / jnp.sum(total_weights)
        p_bar.set_description(f"Loss - {loss_so_far:.4f}")
    return total_loss / total_weights, total_weights


def train(
    params: Params,
    model: ApplyFn,
    dataset: Dataset,
    batch_size: int,
    epochs: int,
    opt_state: optax.OptState,
    optimizer: optax.GradientTransformation,
    key: jax.random.KeyArray,
    max_length: int,
    pad_token_id: int,
    steps_per_epoch: Optional[int] = None,
) -> Tuple[Params, optax.OptState]:
    loss_fn = build_loss_fn(model, pad_token_id)

    @jax.jit
    def update(
        params: Params,
        opt_state: optax.OptState,
        key: jax.random.KeyArray,
        batch: jax.Array,
    ) -> Tuple[Params, optax.OptState, jax.Array]:
        loss, grads = jax.value_and_grad(loss_fn)(params, key, batch)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    num_batches = len(dataset) // batch_size
    steps_per_epoch = num_batches if steps_per_epoch is None else steps_per_epoch
    for epoch in range(epochs):
        print(f"Epoch {epoch:{len(str(epochs))}d}/{epochs}")
        p_bar = tqdm(
            dataset.iter(batch_size=batch_size, drop_last_batch=True),
            total=steps_per_epoch,
            desc="Starting",
        )
        for step_no, batch in enumerate(p_bar):
            if step_no == steps_per_epoch:
                break
            processed = pad_and_convert_batch(batch, max_length, pad_token_id)
            key, subkey = jax.random.split(key)
            params, opt_state, this_loss = update(params, opt_state, subkey, processed)
            p_bar.set_description(f"Loss - {this_loss:.4f}")
    return params, opt_state
