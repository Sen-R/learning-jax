import pickle
from typing import Dict, List

import numpy as np
import tensorflow as tf  # type: ignore
from absl import app, flags  # type: ignore

FLAGS = flags.FLAGS


class _GPT2WeightsTranslator:
    def __init__(self, model_dir: str) -> None:
        self.reader = tf.train.load_checkpoint(model_dir)
        self._available_vars: Dict[str, List[int]] = {}

    def translate(self) -> Dict[str, Dict[str, np.ndarray]]:
        self._available_vars = self.reader.get_variable_to_shape_map()
        layer_labels = set(
            v.split("model/h")[1].split("/")[0]
            for v in self._available_vars
            if v.startswith("model/h")
        )

        wpe = self._translate_embedding("model/wpe")
        wte = self._translate_embedding("model/wte")
        ln_final = self._translate_layer_norm("model/ln_f")

        layers = {
            k: v
            for layer in layer_labels
            for k, v in self._translate_transformer_layer(
                from_prefix=f"model/h{layer}", to_prefix=f"layer_{layer}"
            ).items()
        }

        if len(self._available_vars) > 0:
            raise ValueError(
                "Error: not all variables have been used up, "
                f"still have: {self._available_vars}"
            )

        params: Dict[str, Dict[str, np.ndarray]] = {
            "token_embedding": {"w": wte},
            "position_embedding": {"w": wpe},
            "ln_final": ln_final,
            **layers,
        }

        return params

    def _pop_tensor(self, name: str) -> np.ndarray:
        if name not in self._available_vars:
            raise KeyError(name)
        tensor: np.ndarray = self.reader.get_tensor(name)
        self._available_vars.pop(name)
        return tensor.squeeze()

    def _pop_w_and_b(self, prefix: str) -> Dict[str, np.ndarray]:
        w = self._pop_tensor(f"{prefix}/w")
        b = self._pop_tensor(f"{prefix}/b")
        return {"w": w, "b": b}

    def _translate_embedding(self, name: str) -> np.ndarray:
        return self._pop_tensor(name)

    def _translate_layer_norm(self, prefix: str) -> Dict[str, np.ndarray]:
        return {
            "offset": self._pop_tensor(f"{prefix}/b"),
            "scale": self._pop_tensor(f"{prefix}/g"),
        }

    def _translate_transformer_layer(
        self, from_prefix: str, to_prefix: str
    ) -> Dict[str, Dict[str, np.ndarray]]:
        params: Dict[str, Dict[str, np.ndarray]] = {}

        # Layer norm parameters
        params[f"{to_prefix}/ln_pre_mha"] = self._translate_layer_norm(
            f"{from_prefix}/ln_1"
        )
        params[f"{to_prefix}/ln_pre_ffn"] = self._translate_layer_norm(
            f"{from_prefix}/ln_2"
        )

        # MHA parameters
        qkv_params = self._pop_w_and_b(f"{from_prefix}/attn/c_attn")
        ws = np.split(qkv_params["w"], 3, axis=-1)
        bs = np.split(qkv_params["b"], 3, axis=-1)
        for split, w_split, b_split in zip(("query", "key", "value"), ws, bs):
            params[f"{to_prefix}/mha/{split}"] = {"w": w_split, "b": b_split}
        params[f"{to_prefix}/mha/out"] = self._pop_w_and_b(f"{from_prefix}/attn/c_proj")

        # FFN parameters
        params[f"{to_prefix}/ffn/fc"] = self._pop_w_and_b(f"{from_prefix}/mlp/c_fc")
        params[f"{to_prefix}/ffn/out"] = self._pop_w_and_b(f"{from_prefix}/mlp/c_proj")
        return params


def main(argv: List[str]) -> None:
    """Translates model weights from GPT-2 TF format to JAX."""

    try:
        _, model_dir, output_file, *rest = argv
    except ValueError:
        raise SystemExit(f"Error parsing arguments, got: {argv[1:]}")

    if rest:
        raise SystemExit(f"Extra arguments passed to command: {rest}")

    translator = _GPT2WeightsTranslator(model_dir)
    params = translator.translate()
    with open(output_file, "wb") as f:
        pickle.dump(params, f)


if __name__ == "__main__":
    app.run(main)
