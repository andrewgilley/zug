# Thesis: WebAssembly, Edge ML, and Distributed Runtime Infrastructure

## Core Thesis

WebAssembly can compete with parts of the container infrastructure ecosystem because it provides a smaller, more portable, more sandboxable execution unit for workloads that do not need a full operating-system packaging boundary.

Containers remain important for conventional cloud services and dependency-heavy applications. The stronger argument is narrower: many edge workloads need fast startup, strong isolation, explicit host capabilities, predictable resource limits, and cross-platform deployment more than they need a full container image.

ML inference at the edge is a strong fit for this shift. A model-backed edge workload often contains preprocessing, model invocation, postprocessing, routing, policy, telemetry, and device-specific capability access. WebAssembly can carry the portable application logic while the host runtime controls access to models, tensors, accelerators, storage, networking, and observability.

`zug` applies this thesis directly: it is becoming a WASM runtime and systems layer that understands ML models as first-class runtime objects.

## What The Project Has Already Proven

The project now has enough implementation to make the thesis concrete:

- ONNX models can be decoded, inspected, and executed locally.
- Tensor values are represented explicitly across several data types.
- A growing ONNX operator set can run real small vision models.
- WASM modules can be parsed, validated, instantiated, and interpreted across a meaningful core subset.
- WASI Preview 1 imports support basic guest interaction such as args, environment, time, random, stdin/stdout/stderr, descriptor stat/close/seek, preopen discovery, readonly directory iteration, readonly file open/read/stat, and process exit.
- A WASI-NN shaped ABI can drive host-managed graph execution.
- Guest WASM fixtures can call into the host ML surface.
- Workload manifests can describe a bundle of guest code, model assets, target requirements, and network needs.
- `zug check` can report compatibility for ONNX models, WASM modules, and workload directories.
- `zug agent` can expose node capabilities over real TCP/HTTP.

This is not yet a production runtime, but it is no longer only a concept. The core architecture is now visible in code.

## Why WebAssembly Matters Here

WebAssembly changes the deployment unit. Instead of packaging a process with broad filesystem and operating-system assumptions, a WASM module packages computation and declares host imports.

For edge ML, that distinction is useful:

- The guest can remain portable across devices.
- The host can enforce capability boundaries.
- Model execution can be provided through a stable ABI instead of embedded native dependencies.
- Runtime compatibility can be checked before execution.
- Updates can ship as smaller artifacts.
- Multi-tenant execution on gateways becomes more plausible.
- Placement can account for model, dtype, memory, and network constraints.

This is the basis for a runtime that sits between generic WASM execution and production ML deployment.

## Why Containers Are Not The Whole Answer

Containers are excellent for many server workloads, but they are often too broad for constrained edge inference:

- They assume a heavier host operating-system boundary.
- They tend to hide application requirements inside images.
- They do not naturally expose model/operator compatibility as a scheduling primitive.
- They can be awkward for heterogeneous devices with different accelerators, memory limits, and network policies.
- Their security boundary is broader than a small, capability-oriented guest module.

The opportunity is not to replace containers everywhere. The opportunity is to build a better deployment and execution unit for portable edge inference.

## The Zug Interpretation

The strongest form of `zug` is a model-aware WebAssembly runtime:

- WASM carries application behavior.
- WASI carries basic system interaction.
- WASI-NN carries model inference calls.
- ONNX is the first model backend.
- Tensors and graph execution are explicit runtime concepts.
- Capability checks explain whether a model-backed guest can run on a target.
- Workload manifests become the deployment unit.
- Agents expose target capabilities and eventually accept checked workloads.

This positions the project as an edge inference environment, not just a WASM interpreter and not just an ONNX executor.

## Commercial Implication

The commercial opportunity is a trusted runtime layer for portable edge inference.

Useful product forms include:

- A compatibility CLI for model and workload readiness.
- An embedded runtime SDK for device vendors.
- A lightweight edge agent for capability reporting and controlled execution.
- A workload package format for WASM plus model assets.
- A model-aware scheduler for heterogeneous fleets.
- Diagnostics for unsupported ops, dtypes, memory limits, WASI imports, and network requirements.
- A secure host capability layer for accelerators and device resources.

The first valuable product can be the tool that tells ML engineers whether a model-backed WASM workload can run on a target edge device, why it cannot, and what has to change.

## Technical Implication

The project should keep developing along these lines:

- Harden `zug check` into a serious compatibility product.
- Add `zug run workload/` as the central local execution workflow.
- Keep WASI-NN as the stable ML ABI shape.
- Expand WASI support only where it helps real guests.
- Increase WASM instruction and validation coverage through spec tests.
- Improve ONNX op, dtype, shape, and attribute coverage.
- Optimize execution through planning, buffer reuse, and specialized kernels.
- Treat networking as explicit host authority, not ambient guest access.
- Grow the agent from capability reporting to checked workload execution.

## Strategic Claim

If edge AI continues moving toward heterogeneous local execution, then runtime infrastructure needs to reason about more than processes and containers. It needs to reason about models, tensors, accelerators, memory limits, WASM imports, network policy, privacy, and placement.

`zug` should become the layer that connects those concerns into a portable execution environment.
