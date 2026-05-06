# Next Project Directions

## Immediate Recommendation

The project should now focus on one clear vertical MVP:

```text
workload directory -> compatibility check -> local run -> agent check -> agent run -> telemetry
```

The core idea is to prove that a model-backed WASM workload can be checked, executed locally, sent to an edge agent, run through a stable WASI-NN shaped ABI, execute an ONNX model, and emit useful operational telemetry.

The project already has many of the pieces needed for this path:

- ONNX model loading and execution.
- Tensor representation and operator coverage.
- A custom WASM parser, validator, instance model, and interpreter.
- WASI Preview 1 host imports.
- WASI-NN shaped inference imports.
- Workload manifests and target profiles.
- Compatibility checks for models, modules, and workloads.
- An HTTP agent with capabilities and telemetry surfaces.
- Early GPU host extension scaffolding.
- Robotics deployment resources.

The main risk now is spreading development across too many surfaces before the central runtime workflow feels complete.

## 1. Keep `zug run workload/` As The Central Workflow

This is now the main local integration point.

The command should:

- Load `zug.toml`.
- Resolve the guest WASM module and model artifact.
- Validate target profile, memory, dtype, model encoding, WASI-NN needs, and network requirements.
- Instantiate the WASM guest.
- Bind WASI and WASI-NN host imports.
- Preload or load the model.
- Call the configured guest entrypoint.
- Return structured outputs and structured errors.
- Emit telemetry for each major runtime phase.

This turns the project from a set of working subsystems into a coherent product behavior.

## 2. Promote The Agent Into An Edge Node Runtime

The agent already exposes health, capabilities, telemetry, and OTLP-oriented surfaces. The next step is to make it useful as an edge node in a distributed system.

Near-term agent work:

- Add `POST /workloads/check`.
- Return the same compatibility details as `zug check workload/`.
- Add persistent node and workload state.
- Add request IDs.
- Add structured errors.
- Add request limits.
- Add token or mTLS-ready authentication.
- Add `POST /workloads/run` only after local `zug run workload/` is stable.

The agent should become the controlled edge execution boundary, not just a capability server.

## 3. Harden WASM Runtime Correctness

The custom runtime has meaningful coverage, but production credibility depends on correctness and predictable failure modes.

Important work:

- Ingest official WASM spec tests.
- Track pass/fail coverage by instruction group.
- Strengthen validation semantics.
- Improve trap reporting with function indexes and instruction offsets.
- Add fuel, timeout, and interruption support.
- Enforce memory and resource limits.
- Broaden SIMD execution support.
- Keep compatibility reports precise for unsupported features.

This is necessary before positioning the runtime against mature WASM engines.

## 4. Improve ONNX Execution For Model Families

The executor has useful operator coverage and can run important smoke tests, but modern edge ML needs stronger model-family support.

Priority work:

- Add shape inference passes.
- Improve dtype behavior.
- Expand quantized operator support.
- Add dynamic shape handling where practical.
- Add execution planning with numeric tensor slots.
- Reuse scratch buffers across runs.
- Specialize Conv kernels for depthwise and pointwise convolution.
- Add stronger reference-output tests against ONNX Runtime.
- Maintain model-family gates for MobileNet, YOLO-style detection, and small transformer-style graphs.

The project should move from "can run some models" toward "can reliably support known edge model families."

## 5. Turn GPU Support Into A Real Backend Strategy

The current GPU host extension is useful scaffolding. The next stage is choosing and implementing a real backend path.

Options to evaluate:

- Vulkan.
- CUDA.
- ROCm.
- wgpu-native or Dawn.
- A layered backend abstraction with CPU fallback.

Needed capabilities:

- Device discovery.
- Queue selection.
- Buffer allocation.
- Tensor buffer interop.
- Kernel or shader dispatch.
- Capability reporting through the agent.
- Placement decisions based on accelerator availability.

Accelerator targeting is central to the commercial edge ML story.

## 6. Make Telemetry Operational

Telemetry should move from a general agent feature into the runtime's operational nervous system.

Runtime spans and metrics should cover:

- Workload compatibility checks.
- Manifest loading.
- WASM parsing.
- WASM validation.
- WASM instantiation.
- WASI import calls.
- WASI-NN graph loading.
- Tensor input binding.
- Model execution.
- Output collection.
- Agent request handling.
- Compatibility failures.
- Runtime traps.

This gives the project the observability layer needed for fleet deployment and debugging.

## 7. Use Robotics As The First Demanding Vertical

Robotics and industrial operations are a strong application area because they stress the runtime in useful ways:

- Model inference affects real-world actions.
- Safety boundaries matter.
- Edge devices are heterogeneous.
- Offline replay and staged rollout are important.
- Telemetry and auditability are required.

The next robotics-specific runtime concept should be an action envelope:

- Guest WASM proposes an action.
- Host validates the proposal.
- Policy metadata records the allowed operating envelope.
- Telemetry records proposal, validation, and outcome.
- Final actuation remains outside the guest and behind independent safety controls.

This keeps `zug` positioned as a policy and inference runtime, not as a certified robot controller.

## Recommended Build Order

1. Add runtime telemetry around the local workload path.
2. Add repeated-run stability tests for `zug run workload/`.
3. Add `POST /workloads/check` to the agent.
4. Add agent request IDs, structured errors, request limits, and basic auth.
5. Add `POST /workloads/run`.
6. Add official WASM spec-test ingestion.
7. Expand ONNX conformance tests for MobileNet, YOLO-style detection, and a small transformer-style model.
8. Add ONNX execution planning and scratch-buffer reuse.
9. Turn the GPU host extension into a real backend abstraction.
10. Add robotics action envelopes and deployment evidence hooks.

## Product Focus

The strongest near-term product shape is:

```text
zug check workload/
zug run workload/
zug agent --node edge-a --profile vision-f32-basic
```

The commercial wedge is not a generic WASM runtime. It is a model-aware edge runtime that can answer:

```text
Can this WASM plus model workload run on this edge node under these memory,
dtype, operator, import, accelerator, network, latency, and policy constraints?
```

That question connects the ONNX executor, WASM runtime, WASI-NN ABI, workload manifests, target profiles, agent, telemetry, GPU surface, and robotics deployment resources into one coherent project direction.
