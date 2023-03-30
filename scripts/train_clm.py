import logging
import os
from functools import partial
from typing import Dict, List, Protocol

import haiku as hk
import jax
import jax.numpy as jnp
import optax  # type: ignore
from absl import app, flags  # type: ignore
from datasets import DatasetDict, load_dataset  # type: ignore
from transformers import AutoTokenizer, GPT2TokenizerFast  # type: ignore

from learningjax import causal_lm, transformer

log_level = os.getenv("LOGLEVEL", logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(log_level)
logging.getLogger("jax").setLevel(log_level)


FLAGS = flags.FLAGS
flags.DEFINE_string("model_name", "", "Model name.")
flags.DEFINE_integer("max_length", 1024, "Max length of sequences.", lower_bound=1)
flags.DEFINE_float("learning_rate", 0.1, "Adam learning rate.", lower_bound=0.0)
flags.DEFINE_integer("batch_size", 32, "Batch size.", lower_bound=0)
flags.DEFINE_integer("epochs", 1, "Number of epochs.", lower_bound=1)
flags.DEFINE_integer("seed", 0, "Random seed.")
flags.DEFINE_integer("embed_dim", 768, "Model dimension.", lower_bound=1)
flags.DEFINE_integer("num_layers", 12, "Number of transformer layers.", lower_bound=0)
flags.DEFINE_integer("num_heads", 12, "Number of parallel heads.", lower_bound=1)


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


def build_unigram_model(*, vocab_size: int) -> hk.Transformed:
    @hk.transform
    def unigram_model(x: jax.Array) -> Dict[str, jax.Array]:
        b = hk.get_parameter(
            "b", shape=(vocab_size,), dtype=jnp.float32, init=jnp.zeros
        )
        new_shape = [1] * x.ndim + [vocab_size]
        return {"logits": b.reshape(new_shape)}

    return unigram_model


def build_transformer(*, vocab_size: int) -> hk.Transformed:
    return transformer.build_causal_transformer(
        vocab_size=vocab_size,
        context_size=FLAGS.max_length,
        embed_dim=FLAGS.embed_dim,
        num_layers=FLAGS.num_layers,
        num_heads=FLAGS.num_heads,
    )


class ModelBuilder(Protocol):
    def __call__(self, *, vocab_size: int) -> hk.Transformed:
        ...


_model_registry: Dict[str, ModelBuilder] = {
    "unigram": build_unigram_model,
    "transformer": build_transformer,
}


def prepare_model(name: str, *, vocab_size: int) -> hk.Transformed:
    try:
        model_builder = _model_registry[name]
    except KeyError:
        raise KeyError(f"Unknown model: {name}") from None
    return model_builder(vocab_size=vocab_size)


flags.register_validator(
    "model_name",
    lambda s: s in _model_registry.keys(),
    f"Model name unrecognised, should be one of {list(_model_registry)}.",
)


def main(argv: List[str]) -> None:
    tokenizer: GPT2TokenizerFast = prepare_tokenizer()
    dds: DatasetDict = prepare_dataset("wikitext-2", tokenizer, FLAGS.max_length)
    model = prepare_model(FLAGS.model_name, vocab_size=tokenizer.vocab_size)
    optimizer: optax.GradientTransformation = optax.adam(FLAGS.learning_rate)

    key = jax.random.PRNGKey(FLAGS.seed)
    key, init_key, train_key = jax.random.split(key, 3)
    params = model.init(init_key, jnp.zeros((1, 1), dtype=jnp.int32))
    opt_state = optimizer.init(params)

    print("Model parameter summary")
    print("=======================")
    module_to_shapes_dict = jax.tree_util.tree_map(lambda l: l.shape, params)
    for module, shapes_dict in module_to_shapes_dict.items():
        print(f"{module}: {shapes_dict}")
    print()
    num_params = jax.tree_util.tree_reduce(lambda c, l: jnp.size(l), params, 0)
    print(f"Total parameters: {num_params:,d}")
    print()

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
