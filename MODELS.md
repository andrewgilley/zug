# Model Corpus

This file tracks modern model targets for capability inspection, ONNX op coverage, and eventual end-to-end WASM guest inference tests.

## Tier 1: Edge Vision ONNX

### MobileNetV2 ONNX

- Source: https://huggingface.co/onnxmodelzoo/mobilenetv2-12
- Download hint: https://huggingface.co/onnxmodelzoo/mobilenetv2-12/resolve/main/mobilenetv2-12.onnx
- Why it matters: mobile/embedded image classification baseline with real convolution-heavy structure.
- Expected pressure on Zug: Conv, Add, Relu/Clip-style activations, pooling, reshape/flatten, Softmax.
- Observed operators: Conv x52, Clip x35, Add x10, GlobalAveragePool x1, Shape x1, Constant x1, Gather x1, Unsqueeze x1, Concat x1, Reshape x1, Gemm x1.
- Current Zug coverage: all observed operators are recognized as supported.
- Execution smoke test: `zig build -Doptimize=ReleaseFast run -- models\mobilenetv2-12.onnx --input input=tmp\mobilenetv2-zero.f32`
- Conformance smoke test: `zig build -Doptimize=ReleaseFast run -- models\mobilenetv2-12.onnx --input input=tmp\mobilenetv2-zero.f32 --output output=tmp\mobilenetv2-zero-zug-output.f32 --expect output=tmp\mobilenetv2-zero-ort-output.f32 --tolerance 0.0001`
- Smoke test result: produces `output float32[1,1000]` and matches the ONNX Runtime zero-input reference within `0.0001`.
- Remaining issues: symbolic `batch_size` dimensions are inferred from raw input file size for execution but still reported by capability inspection.
- Status: downloaded at `models/mobilenetv2-12.onnx`.

### SqueezeNet ONNX

- Source: https://huggingface.co/onnxmodelzoo/squeezenet1.0-7
- Why it matters: small CNN baseline designed for size-constrained deployment.
- Expected pressure on Zug: Conv, Relu, MaxPool, Concat, Dropout pass-through or removal, Softmax.
- Status: candidate, not vendored.

### YOLOv8n ONNX

- Source: https://huggingface.co/webml/yolov8n
- Download hint: https://huggingface.co/webml/yolov8n/resolve/main/onnx/yolov8n.onnx
- Why it matters: compact modern object detection graph with convolutional feature extraction, detection-head reshape/split/concat patterns, resize, sigmoid activations, and post-head tensor layout transforms.
- Expected input shape: `images` as float32 `[1,3,640,640]`.
- Observed operators: Conv x64, Sigmoid x58, Mul x60, Constant x22, Split x9, Add x9, Concat x19, MaxPool x3, Resize x2, Reshape x5, Transpose x1, Softmax x1, Shape x1, Gather x1, Div x2, Slice x2, Sub x2.
- Current Zug coverage: all observed operators are recognized as supported.
- Execution smoke test: `zig build run -- models\yolov8n.onnx --input images=tmp\yolov8n-zero.f32 --output output0=tmp\yolov8n-zero-output.f32`
- Smoke test result: produces `output0 float32[1,84,8400]` from a zero-filled input and writes `tmp\yolov8n-zero-output.f32`.
- Remaining issues: this is a zero-input execution smoke test, not an accuracy or ONNX Runtime conformance comparison yet.
- Status: downloaded at `models/yolov8n.onnx`.

### Tiny Random ViT ONNX

- Source: https://huggingface.co/optimum-intel-internal-testing/tiny-random-vit
- Why it matters: small transformer-style vision model suitable for fast compatibility checks.
- Expected pressure on Zug: Reshape, Transpose, MatMul, Add, Div, Softmax, LayerNormalization-family patterns.
- Status: candidate, not vendored.

## Tier 2: Transformer ONNX

### Tiny Random Llama ONNX

- Source: https://huggingface.co/onnx-community/tiny-random-LlamaForCausalLM-ONNX
- Why it matters: tiny language-model graph that exercises transformer patterns without a large model download.
- Expected pressure on Zug: Gather, Shape, Unsqueeze, Slice, Reshape, Transpose, MatMul, Add, Mul, Div, Softmax.
- Status: candidate, not vendored.

## Tier 3: Robotics Policy Targets

### LeRobot ACT Koch Policy

- Source: https://huggingface.co/docs/lerobot/main/inference
- Example policy path: `lerobot/act_koch_real`
- Why it matters: robotics policy deployment is close to the edge-runtime thesis.
- Expected pressure on Zug: model conversion workflow first; likely PyTorch policy artifacts before ONNX runtime testing.
- Status: conversion target, not an ONNX fixture yet.

### Robotics Policy Admission Notes

- A robotics policy model should not be admitted only because it loads. It needs a policy card, safety case, action-space description, rollout evidence, and a host-side action envelope.
- Prefer small policy fixtures first: imitation-learning policies, action-chunking transformer variants, or tiny exported policies that can run in CI.
- Record observation shapes, action shapes, units, coordinate frames, model hash, guest hash, and target hardware.
- Promote a policy to runtime testing only after `zug check workload/ --kind workload --json` reports a clean workload.
- Keep final actuation outside the model runtime and behind independent industrial controls.

## Admission Rules

- Prefer small or tiny models first so CI can run capability inspection quickly.
- Do not vendor large models directly into the repository.
- Add a model only after recording its source URL, license, expected input shape, and observed unsupported ops.
- Use `zig build run -- inspect <model.onnx>` as the first gate.
- Promote a model to runtime testing only after its unsupported op list is understood.
