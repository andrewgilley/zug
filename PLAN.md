# Zug Current Project Plan

## Aim

`zug` is being built as a model-aware WebAssembly runtime for edge inference. The project combines four parts:

- ONNX model loading, compatibility analysis, and CPU execution.
- A custom WebAssembly parser, validator, instance model, and interpreter.
- WASI and WASI-NN shaped host interfaces so guest modules can drive inference.
- Workload manifests, target profiles, and host-side networking for edge deployment.

The goal is not to clone WasmEdge, WAMR, or ONNX Runtime in full. The goal is narrower and more product-focused: run portable WASM application logic that calls a stable ML ABI, then execute models through a controlled host runtime with strong diagnostics for edge targets.

## Current State

`zug` is now beyond a design sketch. It has an early but working local runtime path.

Implemented project surfaces:

- CLI commands for `scope`, `check`, `inspect`, `bench`, direct ONNX execution, direct WASM execution, local workload execution, and `agent`.
- ONNX protobuf decoding through generated bindings.
- A tensor representation with `float32`, `int64`, `int32`, `uint8`, and `bool` storage.
- Direct ONNX graph execution through `Session` and `Executor`.
- Raw tensor input, output writing, expectation checking, and benchmark reporting.
- Capability inspection for ONNX models, WASM modules, and workload directories.
- A WASM module parser for core sections, imports, exports, memories, tables, globals, elements, code, data, and data count.
- WASM validation for the supported MVP surface.
- WASM instantiation with linear memory, exports, imports, globals, tables, data segments, element segments, and start functions.
- An interpreter with substantial scalar instruction coverage, control flow, function calls, imports, memory operations, and table dispatch.
- WASI Preview 1 host support for args, environment, clock, random, fd stat, fd read/write, descriptor close/seek, preopen discovery, readonly directory iteration, readonly path stat/open, and process exit.
- A WASI-NN shaped host and ABI surface for graph loading, execution context creation, input binding, compute, output descriptors, and output reads.
- Guest WASM fixtures that exercise basic execution, WASI logging, and WASI-NN driven inference.
- Workload manifests through `zug.toml`, target profiles, network requirements, `zug check workload`, and `zug run workload`.
- A host-side `zug agent` that listens over TCP/HTTP and serves `health` and `capabilities`.

The strongest proof point is that a WASM guest can drive ONNX inference through the stable host ABI path, and the project can now check and run a workload bundle against a target profile through one local workflow.

## ONNX Coverage

Current recognized ONNX operators include:

```text
Add, AveragePool, BatchNormalization, Cast, Clip, Concat, Constant,
ConstantOfShape, Conv, Div, Equal, Expand, Flatten, Gather, Gemm,
GlobalAveragePool, Greater, Identity, LeakyRelu, Less, MatMul, MaxPool,
Mul, Pad, Pow, Reciprocal, Relu, ReduceMax, ReduceMean, ReduceSum,
Reshape, Resize, Shape, Sigmoid, Slice, Softmax, Split, Sqrt, Sub,
Squeeze, Tanh, TopK, Transpose, Unsqueeze, Where
```

Current model gates:

- `models/tiny_mnist.onnx` is used for WASI-NN and workload flows.
- `models/mobilenetv2-12.onnx` runs as a real CNN classification smoke test and has an ONNX Runtime zero-input comparison path.
- `models/yolov8n.onnx` runs as a YOLO-style object detection smoke test with zero input.

The ONNX executor is useful, but still not production complete. Remaining work includes dynamic shape depth, attribute coverage, external tensor data, broader dtype math, better execution planning, scratch-buffer reuse, optimized kernels, and more reference-output conformance tests.

## WASM Runtime Coverage

The runtime now has a concrete core:

- Binary parsing and validation for the supported module shape.
- Instantiation with one defined linear memory.
- Bounds-checked loads, stores, memory growth, memory copy, and memory fill.
- Functions, locals, globals, blocks, loops, conditionals, branches, returns, select, tables, and `call_indirect`.
- Scalar integer and floating-point arithmetic, comparisons, conversions, reinterpretation, and extension operations.
- Compatibility reporting for unsupported imports, sections, opcodes, memory features, resources, exports, and WASI requirements before execution.

This is still not a production-grade general WASM runtime. Important gaps remain:

- Official wasm spec-test ingestion.
- Full validation semantics across the whole core spec.
- Broader SIMD execution, not only compatibility recognition for selected forms.
- Threads, shared memory, memory64, exceptions, GC, component model, and advanced proposals.
- Robust fuel, timeout, metering, and interruption.
- A mature linker and host capability policy model.

## WASI And WASI-NN State

The project has a useful WASI Preview 1 subset and a stronger WASI-NN oriented surface.

Implemented WASI Preview 1 imports:

- `args_sizes_get`
- `args_get`
- `environ_sizes_get`
- `environ_get`
- `clock_time_get`
- `random_get`
- `fd_close`
- `fd_fdstat_get`
- `fd_filestat_get`
- `fd_prestat_get`
- `fd_prestat_dir_name`
- `fd_read`
- `fd_readdir`
- `fd_seek`
- `fd_write`
- `path_filestat_get`
- `path_open`
- `proc_exit`

Implemented WASI-NN shaped imports:

- `load_graph`
- `init_execution_context`
- `set_input_by_index`
- `compute`
- `get_output_descriptor`
- `get_output`

Implemented host extension:

- `zug_nn.load_preloaded_graph`

The near-term WASI target is not broad POSIX compatibility. It is enough stable host capability surface for model-backed WASM workloads. The current filesystem slice is intentionally controlled: stdin reads, preopen discovery, readonly directory iteration, and readonly file open/read/stat through host-configured resources. The next WASI expansion should focus on file write policy, HTTP-oriented guest networking, and stronger capability restrictions.

## Workloads And Networking

The project now has an early workload unit:

```text
workload/
  guest.wasm
  model.onnx
  zug.toml
```

The manifest records the entrypoint, guest module, model asset, model encoding, required WASI-NN support, memory, dtype, target profile, and network requirements.

Implemented checks:

- `zug check <model.onnx>`
- `zug check <module.wasm>`
- `zug check workload --kind workload`
- `zug check ... --json`

Implemented agent:

```powershell
zug agent --node edge-a --profile vision-f32-basic --listen 127.0.0.1:7070
```

Current endpoints:

- `GET health`
- `GET capabilities`

The agent is the first real networking surface. It is not yet a scheduler, deployer, secure control plane, or remote workload executor.

## Near-Term Priorities

The highest-value next development slices are:

1. Add runtime telemetry spans and metrics around the local `zug run workload` path.
2. Add `POST /workloads/check` to the agent so a remote caller can ask whether a workload is accepted by a node profile.
3. Add `POST /workloads/run` only after the local workload run path is stable under repeated tests.
4. Expand WASI Preview 1 from readonly preopens into file write policy and guest HTTP capability mapping.
5. Ingest official WASM spec tests for the supported instruction surface.
6. Improve ONNX performance with execution planning, tensor slot indexing, scratch-buffer reuse, and optimized Conv variants.
7. Add conformance fixtures for MobileNetV2, YOLOv8n, and a small transformer-style model.
8. Add auth, request limits, structured errors, and request IDs to the agent before treating it as deployment infrastructure.

## Product Direction

The first credible product is a compatibility and execution tool for ML-backed WASM workloads:

```text
zug check workload
zug run workload
zug agent --node edge-a --profile vision-f32-basic
```

The commercial wedge is not "another generic WASM runtime." It is model-aware edge inference with explicit compatibility diagnostics, a stable WASI-NN shaped ABI, controlled host capabilities, and a path to lightweight distributed deployment.
