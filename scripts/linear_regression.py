import chex
import jax
import learningjax.linear_regression as lr
from learningjax.losses import mse
from learningjax.optimize import sgd
from learningjax.utils import key_generator

SEED = 8293432
N_SAMPLES = 1000
W_TRUE = 3.0
B_TRUE = -2.0
NOISE_SCALE = 0.1
STEPS = 500
ALPHA = 0.01


batched_predict = jax.vmap(lr.linear_model, in_axes=(None, 0))


def compute_loss(
    params: lr.LinearModelParameters, X: chex.Array, y: chex.Array
) -> chex.Array:
    y_pred = batched_predict(params, X)
    return mse(y, y_pred)


@jax.jit
def update(
    params: lr.LinearModelParameters, X: chex.Array, y: chex.Array
) -> lr.LinearModelParameters:
    grads = jax.grad(compute_loss)(params, X, y)
    params = sgd(grads, params, ALPHA)
    return params


def main() -> None:
    key_it = key_generator(seed=SEED)

    true_params = lr.LinearModelParameters(w=W_TRUE, b=B_TRUE)
    X, y = lr.create_dataset(key_it, N_SAMPLES, true_params, noise_scale=NOISE_SCALE)

    params = lr.LinearModelParameters.initialize(key_it, (), ())

    for step in range(STEPS):
        params = update(params, X, y)
        if step % 10 == 0:
            loss = float(compute_loss(params, X, y))
            print(f"Step: {step:4d}/{STEPS:4d} | Loss: {loss:0.4f}")

    print(f"\nTraining complete. Final parameters: {params}")


if __name__ == "__main__":
    main()
