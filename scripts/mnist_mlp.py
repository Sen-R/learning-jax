from typing import Dict, Iterator, List, Tuple

import chex
import jax
import jax.numpy as jnp
import optax  # type: ignore
from absl import app, flags  # type: ignore
from datasets import load_dataset  # type: ignore
from tqdm import tqdm

import learningjax.linear_regression as lr
from learningjax import utils

FLAGS = flags.FLAGS

flags.DEFINE_integer("seed", 0, "Seed to initialize PRNGKey.")
flags.DEFINE_float("learning_rate", 0.001, "Optimizer learning rate.")
flags.DEFINE_integer("batch_size", 128, "SGD batch size.")
flags.DEFINE_integer("epochs", 5, "Number of training epochs.")


def init_params(key_it: Iterator[chex.PRNGKey]) -> lr.LinearModelParameters:
    return lr.LinearModelParameters.initialize(key_it, (28 * 28,), (10,))


def to_logits(params: lr.LinearModelParameters, image: chex.Array) -> chex.Array:
    features = jnp.ravel(image)
    logits = lr.linear_model(params, features)
    return logits


def predict(params: lr.LinearModelParameters, image: chex.Array) -> chex.Array:
    logits = to_logits(params, image)
    return jnp.argmax(logits)  # type: ignore


batch_to_logits = jax.vmap(to_logits, in_axes=(None, 0))
batch_predict = jax.vmap(predict, in_axes=(None, 0))


def loss(params: lr.LinearModelParameters, batch: Dict[str, chex.Array]) -> chex.Array:
    logits = batch_to_logits(params, batch["image"])
    x_ents = optax.softmax_cross_entropy_with_integer_labels(logits, batch["label"])
    return jnp.mean(x_ents)


def accuracy(
    params: lr.LinearModelParameters, batch: Dict[str, chex.Array]
) -> chex.Array:
    return jnp.mean(batch_predict(params, batch["image"]) == batch["label"])


def main(argv: List[str]) -> None:
    jnp.array(0.0)  # To start jax and get warning messages out of the way
    print("\n\n")

    print("Training a MLP on MNIST")
    print("=======================\n")
    print("Loading data:")
    ds = load_dataset("mnist").with_format("jax")
    print("")
    print("Training set size:", len(ds["train"]))
    print("Test set size:    ", len(ds["test"]))
    print("Features shape:   ", ds["train"][0]["image"].shape)
    print("Num classes:      ", 1 + jnp.max(ds["train"]["label"]))
    print("")
    print("Initializing MLP parameters. Initial accuracy: ", end="")
    key_it = utils.key_generator(FLAGS.seed)
    params = init_params(key_it)
    print(f"{accuracy(params, ds['test']):3.2%}")
    print("")

    optimizer = optax.adam(learning_rate=FLAGS.learning_rate)
    opt_state = optimizer.init(params)

    @jax.jit
    def update(
        params: lr.LinearModelParameters,
        opt_state: optax.OptState,
        batch: Dict[str, chex.Array],
    ) -> Tuple[lr.LinearModelParameters, optax.OptState]:
        grads = jax.grad(loss)(params, batch)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state

    print("Starting training...")
    n_epochs: int = FLAGS.epochs
    epochs_width = len(str(n_epochs))
    for epoch in range(1, FLAGS.epochs + 1):
        print(f"Epoch: {epoch:{epochs_width}d}/{FLAGS.epochs:{epochs_width}d}")
        for batch in tqdm(ds["train"].iter(batch_size=FLAGS.batch_size)):
            params, opt_state = update(params, opt_state, batch)
        print(f"Accuracy: {accuracy(params, ds['test']):3.2%}")


if __name__ == "__main__":
    app.run(main)
