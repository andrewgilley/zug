# Distributed WebAssembly Systems For Edge ML

## System Vision

`zug` is evolving into a distributed runtime layer for edge ML workloads. The system should make a WASM module plus model artifact behave like a portable, capability-checked deployment unit.

The long-term goal is a model-aware runtime that can participate in distributed systems the way container runtimes participate in container orchestration, but with a workload model designed for constrained devices, explicit host capabilities, and portable inference.

The project now has the first pieces of this system:

- Local ONNX execution.
- Local WASM parsing, validation, instantiation, and interpretation.
- WASI and WASI-NN shaped host imports.
- Workload manifests.
- Target profiles.
- Network requirement analysis.
- A host-side HTTP agent that exposes node capabilities.

## Workload Unit

A `zug` workload is a directory that can contain guest code, model assets, and a manifest:

```text
workload/
  guest.wasm
  model.onnx
  zug.toml
```

Current example:

```toml
name = "tiny-mnist"
entrypoint = "run"
wasm = "../../../zig-out/wasi_nn_full.wasm"
model = "../../../models/tiny_mnist.onnx"
model_encoding = "onnx"

[requires]
wasi_nn = true
memory = "2MiB"
dtype = "float32"
target = "vision-f32-basic"

[network]
input = "local"
output = "stdout"
ingress = "http"
egress = "https"
max_inflight = 4
request_timeout_ms = 5000
```

Implemented today:

```powershell
zug check examples/workloads/tiny-mnist --kind workload
zug check examples/workloads/tiny-mnist --kind workload --json
zug run examples/workloads/tiny-mnist
```

The run command is now the local equivalent of what an edge agent will eventually execute remotely.

## Runtime Responsibilities

The runtime should separate guest portability from host authority.

Guest responsibilities:

- Application logic.
- Preprocessing and postprocessing.
- Calls to host-provided WASI and WASI-NN imports.
- Policy and routing logic that is safe to sandbox.

Host responsibilities:

- WASM parsing, validation, instantiation, and interpretation.
- Linear memory and table management.
- WASI import implementation.
- WASI-NN graph and execution context management.
- ONNX model execution through the first backend.
- Tensor allocation, dtype handling, and output transport.
- Compatibility analysis and structured errors.
- Resource limits.
- Network and storage capability policy.

This separation is central to the project. Guest modules should not receive broad host access by default. They should receive explicit imports and explicit runtime capabilities.

## Node Model

In the distributed system, each edge node runs a `zug` agent.

Current command:

```powershell
zug agent --node edge-a --profile vision-f32-basic --listen 127.0.0.1:7070
```

Current endpoints:

```text
GET /health
GET /capabilities
```

Current capability response includes:

- Runtime name and agent protocol.
- Node ID.
- Target profile.
- Memory limits.
- Model size limits.
- WASI-NN and WASI HTTP flags.
- Supported dtypes.
- Supported network protocols.
- Network concurrency and timeout limits.

This is the first real networking layer. It lets a controller or developer tool discover what a node claims to support before sending work.

## Target Profiles

Current profiles:

- `wasm-basic`
- `wasi-nn-basic`
- `vision-f32-basic`

Profiles describe:

- Default and maximum WASM memory.
- Maximum model size.
- WASI-NN availability.
- Supported tensor dtypes.
- Supported network protocols.
- Network concurrency and timeout limits.

The profile system should become the basis for scheduling. A future scheduler should not only ask whether a node has enough memory. It should ask whether the node supports the model format, ONNX operators, tensor dtypes, WASI imports, network policy, and latency requirements.

## Compatibility Flow

`zug check` should be treated as a product surface, not a developer-only helper.

Current artifact checks:

- ONNX model compatibility.
- WASM module compatibility.
- Workload directory compatibility.

WASM checks include:

- Unsupported imports.
- Unsupported non-function imports.
- Unsupported sections.
- Unsupported opcodes.
- Memory features such as shared memory and memory64.
- Requested export availability.
- Requested memory versus runtime limits.
- WASI requirement support.

Workload checks combine:

- Manifest parsing.
- Target profile resolution.
- Required WASI-NN support.
- Required dtype support.
- Model encoding support.
- WASM compatibility.
- Model compatibility.
- Network requirement compatibility.

This is already a strong foundation for edge placement.

## Networking Architecture

Networking should remain layered.

Implemented layer:

- Host-side TCP/HTTP agent.
- `/health` for liveness.
- `/capabilities` for node capability discovery.
- Manifest-level network requirements and compatibility checks.

Next layer:

```text
POST /workloads/check
```

This endpoint should accept workload metadata or manifest bytes and return the same compatibility report as the CLI. It should not execute code.

Execution layer:

```text
POST /workloads/run
GET /workloads/{id}
```

This layer should only be added after `zug run workload/` exists locally. Remote execution must enforce limits, request IDs, structured errors, cancellation, and bounded inputs.

Later distributed layer:

- Node registry.
- Workload package transfer.
- Model cache reporting.
- Placement decisions.
- Edge-to-edge forwarding.
- Status and logs.
- Secure transport.

Guest networking should be separate from host control-plane networking. A guest should only receive network capabilities explicitly granted by the runtime configuration.

## Placement Model

A useful scheduler for this project should reason about ML-specific constraints:

- Does the node support the required WASM imports?
- Does the node support the required WASI-NN ABI surface?
- Does it support the model encoding?
- Does it support the required ONNX operators?
- Does it support the required tensor dtypes?
- Does it have enough memory for the guest, model, and intermediate tensors?
- Does it have the model cached?
- Are ingress and egress protocols allowed?
- Is public egress allowed?
- Is latency, power, privacy, or throughput the dominant constraint?

This is the difference between generic container placement and model-aware edge placement.

## Robotics And Industrial Operations

Robotics policies are a primary edge use case for this architecture. A policy workload can use WASM for portable task logic and ONNX/WASI-NN for model inference, but its output must be treated as an action proposal until a host-side action envelope validates it.

Project resources for this track live in `ROBOTICS.md` and `resources/robotics/`. They define policy cards, safety case templates, deployment gates, and a current-schema robotics workload manifest template.

## Security And Policy

The system should assume edge devices are heterogeneous and sometimes exposed.

Required policy areas:

- Explicit host imports.
- No ambient filesystem access.
- Preopened directories only when needed.
- No ambient guest networking.
- Request size limits.
- Model size limits.
- Memory limits.
- Execution timeout or fuel.
- Structured audit logs.
- Authentication for agent control endpoints.
- Transport security before remote execution is enabled outside local tests.

The current agent is local-development infrastructure. It should not be treated as a secure remote deployment service until auth, limits, and transport policy exist.

## Commercial Shape

The commercial value is a trusted runtime and diagnostics layer for edge inference.

Near-term product:

- `zug check` for model-backed workload readiness.
- Compatibility reports for CI.
- Local workload execution.
- Edge node capability reporting.

Mid-term product:

- Embedded runtime SDK.
- Edge agent.
- Workload package format.
- Remote workload checks.
- Controlled remote workload execution.

Long-term product:

- Model-aware scheduler.
- Fleet control plane.
- Observability and update management.
- Device and accelerator targeting.

The system becomes valuable when it can answer and act on this question:

```text
Can this WASM plus model workload run on this edge node, under these resource,
network, latency, and privacy constraints?
```

## Strategic Goal

The strategic goal is to prove that WebAssembly can serve as a practical systems substrate for distributed ML workloads at the edge.

For `zug`, that means becoming the layer that understands how portable guest code, model artifacts, host capabilities, device constraints, and networks come together to make edge inference deployable.
