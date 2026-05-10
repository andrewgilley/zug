# Component Strategy

## Summary

WebAssembly is most useful for `zug` when it is framed as a component and capability substrate, not merely as a smaller container format.

The core strategy is this:

```text
zug should coordinate model-backed edge workloads through typed, sandboxed
capabilities rather than containerized network services.
```

This direction fits the project's current shape. `zug` already has a WASM runtime, WASI and WASI-NN shaped host imports, workload manifests, target profiles, network policy checks, an edge agent, telemetry, and a WIT file for edge inference. The next architectural step is to make those pieces feel like one capability-linked runtime.

The goal is not to clone wasmCloud or implement the full WebAssembly Component Model immediately. The goal is to make small portable guests declare the interfaces they need, then let the host or agent bind those interfaces to controlled providers.

## Strategic Claim

Containers package operating environments. Wasm components package behavior behind typed interfaces.

That distinction matters for edge inference because a `zug` workload is not just a process. It is a bundle of:

- Guest WASM application logic.
- One or more model artifacts.
- WASI and WASI-NN requirements.
- Tensor dtype and memory requirements.
- Network and telemetry requirements.
- Target profile constraints.
- Host-controlled capability policy.

`zug` should make those requirements explicit and checkable before execution. The product value is the ability to answer:

```text
Can this WASM plus model workload run on this edge node, with these host
capabilities, under these memory, dtype, operator, network, latency, and policy
constraints?
```

## Key Improvements

### 1. Make WIT The Canonical Contract

The existing `wit/edge-inference.wit` file should become more than reference documentation. It should define the intended typed boundary between guest modules and host capabilities.

Near-term improvements:

- Treat the `edge-guest` world as the canonical guest contract.
- Have `zug check` report whether a guest's imports and exports match the intended world.
- Have agent capability JSON advertise supported worlds and interfaces.
- Keep new guest-facing APIs expressed first as typed capabilities, then map them to host imports or runtime internals.

This gives `zug` a standards-aligned interface story without requiring full Component Model execution on day one.

### 2. Extend Manifests From Requirements To Bindings

Current workload manifests describe requirements such as memory, dtype, WASI-NN, target profile, and network protocols. The next step is to describe how guest imports are satisfied by host providers.

Future manifests should support a capability binding section similar to:

```toml
[capabilities]
inference = "host.onnx"
diagnostics = "host.compatibility"
telemetry = "agent.otlp"
log = "agent.log"
network.egress = "host.https"
```

The exact schema can evolve, but the behavior should be clear:

- A workload declares the capabilities it needs.
- A target profile or agent declares the providers it offers.
- `zug check workload --json` reports which bindings are satisfied, missing, or rejected by policy.
- Execution only proceeds when required bindings are satisfied.

This moves compatibility checking from "does the runtime support this import?" toward "which explicit provider is authorized to satisfy this interface?"

### 3. Turn The Agent Into A Capability Node

The agent is already more than a health endpoint. It exposes capabilities, workload checks, package transfer, workload state, jobs, and telemetry. The component strategy should make the agent the node-local capability authority.

Near-term agent direction:

- `/capabilities` should include supported worlds, interfaces, providers, model backends, network providers, telemetry providers, and accelerator providers.
- `/workloads/check` should return a binding and placement report, not only a pass/fail compatibility result.
- `/workloads/deploy` should accept only workloads whose required capabilities can be satisfied under the node profile and runtime policy.
- `/workloads/{id}/invoke` should execute through the same checked bindings used during deployment.

This creates a small `zug`-native version of the "lattice" idea without needing distributed runtime linking yet.

### 4. Treat Networking As A Host Capability

`zug` already models network requirements and target profiles. The next improvement is to keep guest networking separate from raw socket access.

Guests should not receive ambient networking. They should import specific capabilities such as:

- HTTP client.
- HTTP ingress handler.
- Message publish.
- Message subscribe.
- Telemetry emit.
- Storage read/write.

The host can then bind those capabilities to concrete providers such as local, stdio, HTTP, HTTPS, NATS, MQTT, or a disabled provider. That keeps the programming model stable while allowing deployment-specific transport choices.

This is the practical architectural difference from container networking. The guest asks for a typed capability, not an IP address, socket, sidecar, or service mesh route.

### 5. Use Components To Sharpen Product Positioning

The project should avoid a broad claim that Wasm replaces containers. The stronger claim is narrower and more useful:

```text
zug replaces container-heavy deployment patterns for model-backed edge inference
workloads that benefit from fast startup, sandboxing, explicit host capabilities,
portable guest logic, and model-aware placement.
```

This gives the project a clearer product identity:

- Not a generic WASM interpreter.
- Not a generic ONNX executor.
- Not a Kubernetes replacement.
- A model-aware edge runtime that binds portable guest logic to controlled host capabilities.

### 6. Apply The Model To Robotics

Robotics policies are a strong vertical for this strategy because model output can influence real-world action.

The component framing suggests a clear runtime boundary:

```text
sensor input -> WASM policy guest -> WASI-NN inference -> action proposal
    -> host action validator -> controller adapter -> independent safety layer
```

The guest should not actuate hardware directly. It should produce an action proposal through a typed capability. The host validates that proposal through an action envelope before anything reaches hardware.

Useful robotics capabilities:

- `observation-input`
- `policy-inference`
- `action-proposal`
- `action-validator`
- `telemetry`
- `audit-log`

This makes robotics support native to the runtime architecture instead of only documentation around it.

## Recommended Build Order

1. Add world/interface awareness to `zug check` reports.
2. Add supported worlds, interfaces, and providers to agent `/capabilities`.
3. Add capability binding fields to workload manifests.
4. Add binding diagnostics to `zug check workload --json`.
5. Make `/workloads/check` return the same binding diagnostics remotely.
6. Require successful bindings before deploy or invoke.
7. Keep network access provider-backed rather than ambient.
8. Add robotics action-envelope bindings after the local workload and agent paths are stable.
9. Evaluate full Component Model execution after the current WASM/WASI-NN runtime path is reliable.

## Non-Goals

This strategy does not require immediate support for:

- Full WebAssembly Component Model execution.
- Dynamic linking between arbitrary remote components.
- A NATS-backed lattice.
- Replacing Kubernetes for general-purpose cloud services.
- Exposing raw guest networking.
- Rewriting the current WASI-NN ABI before the local runtime path is stable.

The first useful step is smaller: make the existing runtime, manifest, target profile, and agent surfaces describe typed capabilities and explicit bindings.

## Success Criteria

The component strategy is working when `zug` can produce a report like:

```text
workload: tiny-mnist
world: zug:edge-inference/edge-guest@0.1.0
required interfaces:
  inference -> satisfied by host.onnx.cpu
  diagnostics -> satisfied by host.compatibility
  log -> satisfied by agent.log
network:
  ingress http -> allowed
  egress https -> allowed
model:
  encoding onnx -> supported
  dtype float32 -> supported
  operators -> supported
placement:
  target vision-f32-basic -> accepted
```

That is the intended product shape: not just running Wasm, but coordinating lightweight, sandboxed, model-backed workloads through typed host capabilities.
