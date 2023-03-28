import logging
import os
from functools import partial
from typing import Any, Dict, List

import haiku as hk
import jax
import jax.numpy as jnp
import optax  # type: ignore
from absl import app, flags  # type: ignore
from datasets import DatasetDict, load_dataset  # type: ignore
from transformers import AutoTokenizer, GPT2TokenizerFast  # type: ignore

from learningjax import causal_lm

log_level = os.getenv("LOGLEVEL", logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(log_level)
logging.getLogger("jax").setLevel(log_level)


FLAGS = flags.FLAGS
flags.DEFINE_integer("max_length", 1024, "Max length of sequences.", lower_bound=0)
flags.DEFINE_float("learning_rate", 0.1, "Adam learning rate.")
flags.DEFINE_integer("batch_size", 32, "Batch size.", lower_bound=0)
flags.DEFINE_integer("epochs", 1, "Number of epochs.", lower_bound=1)
flags.DEFINE_integer("seed", 0, "Random seed.")


dataset_library = {
    "wikitext-2": ("wikitext", "wikitext-2-raw-v1"),
}


def tokenize_and_chunk(
    example: Dict[str, List[List[int]]], tokenizer: GPT2TokenizerFast, max_length: int
) -> Dict[str, List[List[int]]]:
    tokenized = tokenizer(example["text"])["input_ids"]
    chunks = []
    for row in tokenized:
        chunks += [row[i : i + max_length] for i in range(0, len(row), max_length)]
    return {"input_ids": chunks}


def prepare_tokenizer() -> GPT2TokenizerFast:
    logger.info("Preparing tokenizer")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def prepare_dataset(
    name: str, tokenizer: GPT2TokenizerFast, max_length: int
) -> DatasetDict:
    logger.info(f"Preparing dataset `{name}`")
    raw_dds = load_dataset(*dataset_library[name])
    map_fn = partial(tokenize_and_chunk, tokenizer=tokenizer, max_length=max_length + 1)
    return raw_dds.map(
        map_fn, batched=True, remove_columns=raw_dds["train"].column_names
    )


def build_unigram_model(*, tokenizer: GPT2TokenizerFast) -> hk.Transformed:
    vocab_size = tokenizer.vocab_size

    @hk.transform
    def unigram_model(x: jax.Array) -> Dict[str, jax.Array]:
        b = hk.get_parameter(
            "b", shape=(vocab_size,), dtype=jnp.float32, init=jnp.zeros
        )
        new_shape = [1] * x.ndim + [vocab_size]
        return {"logits": b.reshape(new_shape)}

    return unigram_model


_model_registry = {"unigram": build_unigram_model}


def prepare_model(name: str, **kwargs: Any) -> hk.Transformed:
    try:
        model_builder = _model_registry[name]
    except KeyError:
        raise KeyError(f"Unknown model: {name}") from None
    return model_builder(**kwargs)


def main(argv: List[str]) -> None:
    tokenizer = prepare_tokenizer()
    dds = prepare_dataset("wikitext-2", tokenizer, FLAGS.max_length)
    model = prepare_model("unigram", tokenizer=tokenizer)
    optimizer = optax.adam(FLAGS.learning_rate)

    key = jax.random.PRNGKey(FLAGS.seed)
    key, init_key, train_key = jax.random.split(key, 3)
    params = model.init(init_key, jnp.zeros((1, 1), dtype=jnp.int32))
    opt_state = optimizer.init(params)

    causal_lm.train(
        params,
        model.apply,
        dds["train"],
        FLAGS.batch_size,
        FLAGS.epochs,
        opt_state,
        optimizer,
        train_key,
        FLAGS.max_length,
        tokenizer.pad_token_id,
        val_dataset=dds["test"],
    )


if __name__ == "__main__":
    try:
        app.run(main)
    except KeyboardInterrupt:
        raise SystemExit("KeyboardInterrupt received, exiting.") from None
