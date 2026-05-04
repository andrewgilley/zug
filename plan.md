# WASI-NN Edge ML Runtime Roadmap

## Summary

- Current state: `zug` is a working ONNX execution prototype, not yet a real WASM edge ML runtime.
- Strengths: ONNX protobuf decoding, capability inspection, tensor storage, simple graph execution, and a backend-neutral session shape.
- Gaps: no actual WASM module loading/execution, no WASI-NN host API, narrow ONNX op coverage, no performance story, no hardware acceleration, no packaging/deployment layer.
- Commercial direction: build an ML-focused WASM runtime for ML engineers, centered on WASI-NN-compatible model execution and compatibility diagnostics.

## Current Reality

- ONNX side: usable prototype. It can inspect and run a tiny ONNX graph with `Flatten`, `Gemm`, `Softmax`, plus early support for `Add`, `Conv`, `MatMul`, `Constant`.
- Tensor side: better than before, with basic non-`f32` storage, but most math ops still execute only `float32`.
- WASM side: only a design placeholder exists in `Session`; there is no module parser, validator, linker, memory model, import/export binding, WASI, or WASI-NN implementation yet.
- Commercial readiness: early R&D. Credible demo is reachable; production edge runtime is still a major system build.

## Key Direction

- Do not compete head-on as a generic WasmEdge/WAMR clone first. Build a narrower runtime: "run WASM modules that call ML inference through a WASI-NN-compatible host API."
- Make ONNX execution the first backend behind that API. Guest WASM modules should load a graph, set tensor inputs, compute, and read outputs through runtime-managed graph/context handles.
- Keep `Session` as the public runtime abstraction, but expand it around backend selection, tensor descriptors, graph handles, and execution contexts.
- Treat `capabilities` as a product feature: ML engineers should be able to ask "will this model run on this target runtime/device, and why not?"

## Implementation Milestones

- Milestone 1: WASI-NN host surface
  - Add a `wasi_nn` backend enum/state layer.
  - Define host functions matching the WASI-NN shape: load graph, init execution context, set input, compute, get output.
  - Route graph execution to the existing ONNX executor first.
  - Support only CPU `float32` initially, with explicit unsupported errors.

- Milestone 2: Minimal WASM hosting
  - Add WASM module loading, import resolution, linear memory access, and exported function invocation.
  - Start with core WASM modules, not full Component Model support.
  - The first guest module should call the WASI-NN-like imports and receive ONNX inference output.

- Milestone 3: Model compatibility depth
  - Expand ONNX op coverage toward real vision models: `Reshape`, `Transpose`, `Concat`, `Slice`, `Sigmoid`, `Mul`, `Div`, `Sub`, `MaxPool`, `Resize`, `Shape`, `Gather`, `Cast`, `Unsqueeze`, `Squeeze`.
  - Add op-level dtype rules instead of only model-level dtype support.
  - Add golden tests using exported ONNX models, not only hand-built tensors.

- Milestone 4: Commercial runtime features
  - Add deterministic memory limits, model size limits, execution time limits, and structured errors.
  - Add compatibility reports that name unsupported ops, attributes, dtypes, dynamic shapes, and target-device constraints.
  - Add benchmarks against ONNX Runtime/WasmEdge/WAMR for small vision models.

## Test Plan

- Unit tests for each op and dtype path.
- Golden ONNX tests comparing outputs against known fixtures.
- WASI-NN host tests using a tiny guest WASM module that loads, computes, and reads output.
- Capability tests for unsupported ops, unsupported dtypes, dynamic dimensions, external data, and multi-output nodes.
- Edge-oriented tests for memory limits, malformed model bytes, malformed guest modules, and repeated session execution.

## Assumptions

- "Wasm runtime" means an ML-focused WASM runtime, not a fully general Wasmtime/WAMR replacement on day one.
- First commercial user is an ML engineer who needs model compatibility, runtime diagnostics, and portable edge inference.
- First WASM API target is WASI-NN-compatible behavior because WASI-NN is designed around model loading, graph execution, and host-provided acceleration.
- Relevant ecosystem anchors: [WASI-NN proposal](https://github.com/WebAssembly/wasi-nn), [WASI interfaces](https://wasi.dev/interfaces), [WasmEdge WASI-NN plugin](https://wasmedge.org/docs/contribute/source/plugin/wasi_nn), [WAMR](https://bytecodealliance.github.io/wamr.dev/), and [ONNX Runtime Web/WebGPU](https://onnxruntime.ai/docs/tutorials/web/ep-webgpu.html).
