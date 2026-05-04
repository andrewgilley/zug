# Models

## `tiny_mnist.onnx`

Small MNIST-shaped ONNX model for parser and inference development.

- Input: `image`, `float32[1, 1, 28, 28]`
- Output: `probabilities`, `float32[1, 10]`
- Graph: `Flatten -> Gemm -> Softmax`
- Initializers: `fc.weight`, `fc.bias`

The weights are deterministic seven-segment-style digit templates, not weights
trained on MNIST. This keeps the file self-contained and useful for validating
ONNX graph decoding, tensor loading, and basic execution paths.

Regenerate it with:

```powershell
python tools\generate_tiny_mnist_onnx.py
```

Run it through the current app with:

```powershell
zig build run -- models\tiny_mnist.onnx
```
