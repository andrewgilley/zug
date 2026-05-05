# WASI Positioning

## Position

`zug` should be positioned as a constrained edge inference runtime, not as a general-purpose Wasm runtime clone.

The target is a runtime that can execute small WebAssembly guest modules, expose WASI-compatible host capabilities, and provide a WASI-NN-shaped path into ONNX inference for edge devices.

## Current Runtime Proof

The current proof point is `constrained_edge_inference_flow`:

- Guest code grows memory with `memory.grow`.
- Guest code logs through `wasi_snapshot_preview1.fd_write`.
- Guest code loads `models/tiny_mnist.onnx` through the `wasi_nn` import namespace.
- Guest code initializes an execution context, binds input, calls `compute`, and returns the status.
- The host routes the inference call into the local ONNX executor.

Relevant files:

- `fixtures/constrained_edge_inference_flow.wat`
- `src/wasm/fixtures.zig`
- `src/wasm/interpreter.zig`
- `src/wasm/imports.zig`
- `src/wasi_nn.zig`
- `src/wasi_nn_abi.zig`

## Standards Alignment

WASI 0.2 is based on the WebAssembly Component Model and uses WIT to describe interfaces and worlds. That direction matters for this project because edge inference should be expressed as a capability interface, not only as an ad hoc pointer ABI.

Near term, `zug` should keep its core Wasm ABI path:

- `wasi_snapshot_preview1.fd_write`
- `wasi_snapshot_preview1.proc_exit`
- `wasi_nn.load_graph`
- `wasi_nn.init_execution_context`
- `wasi_nn.set_input_by_index`
- `wasi_nn.compute`
- `wasi_nn.get_output_descriptor`
- `wasi_nn.get_output`

Longer term, every host capability should have a WIT-facing design. The first draft lives in:

- `wit/edge-inference.wit`

That WIT file is a design bridge. It does not replace the current ABI yet. It records the interface shape we want when the runtime grows toward Component Model support.

## Contribution Strategy

The strongest contribution path is around WASI-NN and edge-focused conformance.

`wasi-nn` is still an active proposal, and a useful independent implementation can contribute more than code. This project can produce:

- implementation notes from a Zig host runtime
- test vectors for ONNX graph loading and tensor binding
- failure-mode tests for invalid memory, unsupported dtype, bad handles, and buffer sizing
- edge constraints feedback around model size, memory pages, CPU-only execution, and quantized tensors
- WIT experiments for higher-level inference APIs

## Project Workflow

When adding a new WASI or inference capability:

1. Add or update the current core Wasm ABI in `src/wasm/imports.zig`.
2. Add the runtime behavior in `src/wasm/interpreter.zig` or the host surface.
3. Add a readable `.wat` fixture in `fixtures/`.
4. Add the byte fixture in `src/wasm/fixtures.zig`.
5. Add a runtime test that proves the guest behavior end to end.
6. Update `wit/edge-inference.wit` if the capability belongs in the future Component Model surface.

## Near-Term Roadmap

- Add `get_output_descriptor` and `get_output` to the constrained edge guest flow.
- Add guest-side failure tests for unsupported dtype and invalid handles.
- Add basic args/env WASI calls only if compiled guest toolchains require them.
- Add target profiles to capability reports, starting with `cpu-f32-basic`.
- Add a compatibility report that connects unsupported ONNX ops to target runtime limits.
- Track WASI Component Model and WASI-NN changes without blocking the core MVP on full component support.

## External Anchors

- WASI 0.2 and the Component Model: https://bytecodealliance.org/articles/WASI-0.2
- Component Model concepts: https://component-model.bytecodealliance.org/design/component-model-concepts.html
- WIT reference: https://component-model.bytecodealliance.org/design/wit.html
- WASI resources: https://wasi.dev/resources
- WASI-NN proposal: https://github.com/WebAssembly/wasi-nn
