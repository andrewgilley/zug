#!/usr/bin/env python3
"""Generate a tiny, self-contained ONNX model with MNIST-shaped I/O.

The model is intentionally simple and dependency-light:

    image[1, 1, 28, 28] -> Flatten -> Gemm -> Softmax -> probabilities[1, 10]

The weights are deterministic seven-segment-style digit templates, not weights
from MNIST training. This makes the file useful for parser, graph, tensor, and
basic inference work without needing a network download or training pipeline.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "models" / "tiny_mnist.onnx"


SEGMENTS_BY_DIGIT = {
    0: ("top", "upper_left", "upper_right", "lower_left", "lower_right", "bottom"),
    1: ("upper_right", "lower_right"),
    2: ("top", "upper_right", "middle", "lower_left", "bottom"),
    3: ("top", "upper_right", "middle", "lower_right", "bottom"),
    4: ("upper_left", "upper_right", "middle", "lower_right"),
    5: ("top", "upper_left", "middle", "lower_right", "bottom"),
    6: ("top", "upper_left", "middle", "lower_left", "lower_right", "bottom"),
    7: ("top", "upper_right", "lower_right"),
    8: ("top", "upper_left", "upper_right", "middle", "lower_left", "lower_right", "bottom"),
    9: ("top", "upper_left", "upper_right", "middle", "lower_right", "bottom"),
}


def draw_h(canvas: np.ndarray, y: int, x0: int = 7, x1: int = 21, thickness: int = 3) -> None:
    canvas[y - thickness // 2 : y + thickness // 2 + 1, x0:x1] = 1.0


def draw_v(canvas: np.ndarray, x: int, y0: int, y1: int, thickness: int = 3) -> None:
    canvas[y0:y1, x - thickness // 2 : x + thickness // 2 + 1] = 1.0


def template_for_digit(digit: int) -> np.ndarray:
    canvas = np.zeros((28, 28), dtype=np.float32)
    for segment in SEGMENTS_BY_DIGIT[digit]:
        if segment == "top":
            draw_h(canvas, 5)
        elif segment == "middle":
            draw_h(canvas, 14)
        elif segment == "bottom":
            draw_h(canvas, 23)
        elif segment == "upper_left":
            draw_v(canvas, 6, 6, 14)
        elif segment == "upper_right":
            draw_v(canvas, 21, 6, 14)
        elif segment == "lower_left":
            draw_v(canvas, 6, 15, 23)
        elif segment == "lower_right":
            draw_v(canvas, 21, 15, 23)

    # Light blur so the templates are less brittle if used with rough sketches.
    padded = np.pad(canvas, 1, mode="constant")
    blurred = (
        padded[0:28, 0:28]
        + padded[0:28, 1:29]
        + padded[0:28, 2:30]
        + padded[1:29, 0:28]
        + 4.0 * padded[1:29, 1:29]
        + padded[1:29, 2:30]
        + padded[2:30, 0:28]
        + padded[2:30, 1:29]
        + padded[2:30, 2:30]
    ) / 12.0
    return blurred.astype(np.float32)


def build_model() -> onnx.ModelProto:
    templates = np.stack([template_for_digit(d) for d in range(10)], axis=0)
    flat = templates.reshape(10, 28 * 28)

    norms = np.linalg.norm(flat, axis=1, keepdims=True)
    weights = (flat / norms).T.astype(np.float32)
    bias = np.full((10,), -0.25, dtype=np.float32)

    image = helper.make_tensor_value_info("image", TensorProto.FLOAT, [1, 1, 28, 28])
    probabilities = helper.make_tensor_value_info("probabilities", TensorProto.FLOAT, [1, 10])
    flat_info = helper.make_tensor_value_info("flat", TensorProto.FLOAT, [1, 784])
    logits_info = helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, 10])

    graph = helper.make_graph(
        nodes=[
            helper.make_node("Flatten", ["image"], ["flat"], name="flatten", axis=1),
            helper.make_node("Gemm", ["flat", "fc.weight", "fc.bias"], ["logits"], name="fc"),
            helper.make_node("Softmax", ["logits"], ["probabilities"], name="softmax", axis=1),
        ],
        name="tiny_mnist_template",
        inputs=[image],
        outputs=[probabilities],
        initializer=[
            numpy_helper.from_array(weights, name="fc.weight"),
            numpy_helper.from_array(bias, name="fc.bias"),
        ],
        value_info=[flat_info, logits_info],
    )

    model = helper.make_model(
        graph,
        producer_name="zug",
        producer_version="0.0.1",
        opset_imports=[helper.make_operatorsetid("", 13)],
        doc_string="Tiny MNIST-shaped template classifier for ONNX parser and inference development.",
    )
    model.ir_version = 8
    helper.set_model_props(
        model,
        {
            "dataset": "MNIST-shaped synthetic templates",
            "input": "image float32[1,1,28,28], values usually 0..1",
            "output": "probabilities float32[1,10]",
            "training": "not trained; deterministic seven-segment templates",
        },
    )
    return model


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    model = build_model()
    onnx.checker.check_model(model)
    onnx.save_model(model, OUT)
    print(f"wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
