# Future Roadmap

## Direction

`zug` should continue toward a robust edge inference environment where a workload is a small, portable unit made of:

- Guest WASM application logic.
- One or more model artifacts.
- A manifest describing runtime, model, memory, dtype, network, and target requirements.
- Host-provided capabilities for WASI, WASI-NN, networking, storage, and observability.

The project now has the first versions of those pieces. The future work is to harden them into a coherent runtime product.

## Product Milestones

### Milestone 1: Local Workload Execution

Current state: workload manifests can be checked, but workload execution is still assembled through lower-level commands.

Target:

```powershell
zug run workload/
```

Required work:

- Load `zug.toml`.
- Resolve the guest module and model paths.
- Validate target, memory, dtype, model encoding, and network requirements.
- Instantiate the guest module.
- Bind WASI and WASI-NN host imports.
- Load or preload the model.
- Call the configured entrypoint.
- Return structured outputs and errors.

This is the next major integration point because it converts the project from several working subsystems into one coherent runtime workflow.

### Milestone 2: Agent Workload Check

Current state: `zug agent` listens over TCP/HTTP and exposes node capability JSON through `/health` and `/capabilities`.

Target:

```text
POST /workloads/check
```

Required work:

- Define a request format for workload metadata or manifest bytes.
- Return the same compatibility details as `zug check workload/`.
- Include target profile, unsupported WASM imports, unsupported WASI requirements, unsupported ONNX operators, memory issues, dtype issues, and network-policy issues.
- Keep this endpoint side-effect free.

This turns the agent from a passive capability endpoint into the first useful networked control-plane primitive.

### Milestone 3: Agent Workload Run

Current state: the runtime can execute WASM and ONNX locally, but the agent does not accept remote execution requests.

Target:

```text
POST /workloads/run
GET /workloads/{id}
```

Required work:

- Add a workload ID and lifecycle state.
- Accept bounded workload inputs.
- Enforce memory, model size, and request limits.
- Run the workload through the same path as `zug run workload/`.
- Return output metadata, logs, timing, and structured errors.
- Add cancellation or timeout.

This should come after local workload execution, not before it.

### Milestone 4: Compatibility Product

Current state: `zug check` already reports useful ONNX, WASM, workload, and JSON compatibility details.

Target:

- Make `zug check` the primary developer workflow for model-backed WASM readiness.
- Produce stable machine-readable reports suitable for CI.
- Record exact unsupported operators, attributes, dtypes, dynamic shapes, external data, memory features, imports, exports, and target profile mismatches.
- Add "suggested next fix" fields where possible.
- Keep the report useful even when parsing fails.

This milestone has commercial value before the runtime is production-grade because teams already need to know why a model or module will not run on an edge target.

### Milestone 5: Runtime Production Readiness

Current state: the WASM runtime has meaningful instruction coverage and guest fixtures, but it is not yet a general production runtime.

Required work:

- Import official wasm spec tests and track pass/fail coverage.
- Expand validation semantics.
- Add metering, fuel, timeouts, and interruption.
- Harden import resolution and host capability policy.
- Improve error reporting with function indexes, offsets, import names, and module sections.
- Add fuzzing for module parsing, validation, and compatibility scanning.
- Decide which advanced WASM proposals are explicitly out of scope for the first product.

### Milestone 6: Model Execution Performance

Current state: ONNX execution works for important small vision targets, but performance is still prototype-grade.

Required work:

- Compile ONNX graphs into execution plans.
- Replace repeated string lookups with numeric tensor slots.
- Reuse scratch buffers across runs.
- Specialize Conv for common edge patterns, especially depthwise and pointwise convolution.
- Add shape and dtype inference passes.
- Add reference comparisons for real model fixtures.
- Benchmark Debug, ReleaseSafe, and ReleaseFast profiles separately.

Performance matters because edge runtime value depends on predictable latency, memory use, and startup time.

### Milestone 7: Distributed Edge System

Current state: the agent can advertise capabilities over HTTP, and workload manifests can describe requirements.

Target:

- Node registry.
- Capability-aware placement.
- Workload package transfer.
- Model cache awareness.
- Local network peer discovery.
- Request forwarding between edge nodes.
- Secure transport and authentication.
- Observability across nodes.

The scheduler should reason about more than CPU and memory. It should understand model format, ONNX ops, dtypes, WASI imports, memory limits, network policy, model cache state, and latency constraints.

### Milestone 8: Robotics Policy Deployment

Current state: the project has robotics deployment resources in `ROBOTICS.md` and `resources/robotics/`, but the runtime does not yet have a first-class action envelope.

Target:

- Policy cards attached to robotics workloads.
- Safety case records attached to robotics workloads.
- Host-side action envelope validation.
- Replay, simulation, shadow, advisory, and supervised rollout evidence.
- Agent capability reports that expose whether a node is suitable for robotics policy workloads.

This milestone matters because industrial robotics is where model inference becomes operational action. The runtime must keep policy execution separate from final actuation and preserve independent safety controls.

## Commercial Path

The strongest commercial path is staged:

1. Compatibility CLI for model and workload readiness.
2. Local runtime SDK for embedding model-backed WASM execution.
3. Edge agent for fleet capability reporting and controlled workload execution.
4. Model-aware scheduler for heterogeneous edge environments.
5. Enterprise control plane for deployment, observability, policy, and updates.

The first buyer is likely an ML or edge engineering team that needs to ship models to devices and cannot rely on a uniform cloud/container environment.

## Standards Path

The project should stay aligned with WASI and WASI-NN concepts without blocking progress on every standards change.

Near-term:

- Maintain a WASI-NN shaped ABI that guests can call now.
- Track WASI Preview 1 behavior for practical guest support.
- Add controlled HTTP capabilities in a way that can later map to standard WASI interfaces.

Longer-term:

- Evaluate WASI 0.2 and the Component Model once the core runtime path is stable.
- Keep host capabilities explicit and policy-controlled.
- Use standards alignment as an interoperability strategy, not as a reason to delay the MVP.

## Near-Term Development Order

The recommended order is:

1. Implement `zug run workload/`.
2. Add `POST /workloads/check` to the agent.
3. Add agent request/response JSON tests.
4. Add WASM spec-test ingestion for the currently supported instruction set.
5. Add Conv and execution-plan performance work.
6. Add MobileNetV2 and YOLO conformance tests beyond zero-input smoke tests.
7. Add `POST /workloads/run` once local workload execution is stable.
