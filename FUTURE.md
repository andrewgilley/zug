# Strategic Avenues For A Robust Edge Inference Runtime

## Summary

- The strongest path is to make `zug` an ML-focused WASM runtime, not a generic WasmEdge/WAMR clone.
- The near-term technical goal should be: a guest `.wasm` module calls your WASI-NN-shaped ABI, and that call drives ONNX model inference through your existing executor.
- The commercial wedge should be compatibility, diagnostics, and constrained edge execution for ML engineers shipping models to devices.
- Assumption: the first buyer/user remains ML engineers and edge inference teams, with WASI-NN compatibility as the runtime story.

## Priority Avenues

- **Runtime MVP:** finish the core WASM path from parsed imports/exports to a minimal interpreter that can call WASI-NN imports. This proves the project's central claim: WASM code can drive model loading and inference inside your runtime.
- **Model Compatibility Product:** keep growing `capabilities` into a serious model-readiness report: unsupported ops, dtypes, shapes, attributes, memory needs, and target profile. This can become useful before the runtime is fully production-grade.
- **Edge Execution Discipline:** add memory limits, model size limits, deterministic errors, repeatable execution, and CPU-only profiling. These are the pieces that make the project credible for devices instead of only demos.
- **Backend Strategy:** keep the in-house ONNX executor as the first backend, but design interfaces so optimized backends can be added later. WasmEdge already exposes WASI-NN over many ML backends, so differentiation should come from focused tooling, small footprint, and model diagnostics rather than backend breadth alone.
- **Standards Positioning:** stay WASI-NN-shaped because the official proposal is explicitly about host-provided ML inference, not individual op building. Longer term, track WASI 0.2/Component Model, but do not block the MVP on components.

## Commercial And Investor Path

- **Early commercial wedge:** a CLI/SDK that answers "will this model run on this edge target?" with actionable reports. This is lower risk than selling a full runtime immediately.
- **Developer platform stage:** package WASM + model + target profile into a repeatable deployment artifact. This becomes valuable for teams managing many edge models.
- **Runtime licensing stage:** once the runtime can safely execute guest WASM and enforce resource limits, it can become an embedded SDK for device vendors, robotics, cameras, industrial systems, and privacy-sensitive local inference.
- **Investor narrative:** start with a narrow beachhead: "portable, sandboxed edge inference for ML teams." The broader thesis is that edge AI needs safer deployment containers, compatibility tooling, and local execution as models move closer to devices.
- **Strategic value:** if the project matures into an independent WASI-NN-compatible implementation, it aligns with the direction of the WASI-NN proposal, which is still in Phase 2 and requires independent implementations for later advancement.

## Key Implementation Changes

- Extend the WASM runtime from parsing into instantiation: bind parsed imports to `wasi_nn_abi.Surface`, identify exported memory/functions, and prepare callable function indexes.
- Build the smallest interpreter slice needed for a demo guest: constants, local get/set if needed, calls to imports, calls to exports, return, and end.
- Create one tiny guest WASM fixture whose exported `run` function calls `wasi_nn.load_graph`, `init_execution_context`, `set_input_by_index`, `compute`, and `get_output`.
- Expand capability reports into target profiles such as `cpu-f32-basic`, `vision-f32-basic`, and later `quantized-edge`.

## Test Plan

- WASM parser tests for imports, exports, memories, and code bodies.
- Instance tests that bind a parsed `wasi_nn` import to the ABI resolver.
- Interpreter tests for import calls and basic control flow.
- End-to-end test: guest WASM calls the ABI and runs `models/tiny_mnist.onnx`.
- Compatibility tests using unsupported ops/dtypes/shapes to verify useful diagnostics.

## External Signals

- [WASI-NN](https://github.com/WebAssembly/wasi-nn) is a Phase 2 proposal focused on host-provided ML inference and model loading.
- [WasmEdge WASI-NN](https://wasmedge.org/docs/contribute/source/plugin/wasi_nn) already supports multiple ML backends, so competing on generic backend breadth is hard.
- [WASI 0.2](https://bytecodealliance.org/articles/WASI-0.2) and the [Component Model](https://component-model.bytecodealliance.org/design/component-model-concepts.html) show where standardized WASM interfaces are headed.
- [ONNX Runtime Web/WebGPU](https://onnxruntime.ai/docs/tutorials/web/ep-webgpu.html) shows the high-performance inference bar from mature runtimes.
