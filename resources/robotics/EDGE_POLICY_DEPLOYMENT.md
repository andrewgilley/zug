# Edge Robotics Policy Deployment

## Runtime Architecture

The intended runtime path is:

```text
observation source
  -> WASM policy adapter
  -> WASI-NN model invocation
  -> policy output decoding
  -> host action envelope
  -> controller adapter
  -> independent safety system
```

The WASM policy adapter should stay portable. It should not own broad filesystem, network, or hardware access. The host should provide only the imports and capabilities declared by the workload manifest and target profile.

## Systems Alignment Rules

This deployment model follows `SYSTEMS.md`:

- Workloads are the deployment unit.
- `zug.toml` carries runtime, model, memory, dtype, target, and network requirements.
- `zug check` is the compatibility gate before execution.
- `zug agent` exposes node capability first.
- Agent-side workload checking should come before agent-side workload execution.
- Guest networking is not the same as host control-plane networking.
- Policy output is not final actuation.

## Workload Contents

A robotics policy workload should include:

- `policy.wasm`: guest logic for preprocessing, inference calls, and postprocessing.
- `policy.onnx`: policy, perception, or decision model.
- `zug.toml`: runtime requirements.
- `POLICY_CARD.md`: intended use, action space, observations, limits, and evaluation summary.
- `SAFETY_CASE.md`: deployment evidence and safety review notes.
- Optional calibration files for camera, robot, tool, or cell geometry.
- Optional test fixtures for recorded observations and expected bounded actions.

## Host Responsibilities

The host runtime should be responsible for:

- Loading and validating the workload manifest.
- Checking WASM compatibility before execution.
- Checking model compatibility before execution.
- Enforcing memory and model size limits.
- Binding WASI and WASI-NN imports.
- Restricting network and filesystem access.
- Validating policy inputs and outputs.
- Applying action envelope limits.
- Logging request IDs, policy versions, model hashes, and action decisions.

## Guest Responsibilities

The guest module should be responsible for:

- Transforming observations into model inputs.
- Calling the WASI-NN shaped ABI.
- Transforming model outputs into an action proposal.
- Returning structured status to the host.

The guest should not directly actuate equipment. It should produce proposals that the host can validate.

## Action Envelope

An action envelope is the host-side policy that limits output before a robot or industrial process can receive it.

Typical envelope constraints:

- Joint position bounds.
- Velocity and acceleration bounds.
- Torque or force bounds.
- Cartesian workspace bounds.
- Keep-out zones.
- Tool state limits.
- Maximum action duration.
- Rate limit between consecutive actions.
- Observation age limit.
- Required heartbeat.
- Fallback action.

For the first `zug` implementation, an action envelope can begin as structured metadata and host validation code. Later, it should become part of the workload schema.

## Deployment Stages

### Stage 1: Offline Replay

Run the policy against recorded observations. Verify that outputs are shaped correctly, deterministic when expected, bounded, and logged.

### Stage 2: Simulation

Run the policy in a simulator or digital twin. Verify trajectory quality, failure behavior, and recovery behavior.

### Stage 3: Hardware-In-The-Loop

Connect to real hardware signals without enabling physical actuation. Verify timing, observation freshness, serialization, and controller integration.

### Stage 4: Shadow Mode

Run beside the production controller and log proposed actions without using them.

### Stage 5: Advisory Mode

Allow the policy to recommend actions that require explicit approval by an operator, PLC, or supervisory controller.

### Stage 6: Supervised Bounded Actuation

Enable actuation only inside reduced speed, force, workspace, and task limits.

### Stage 7: Production With Rollback

Allow bounded production use only with monitoring, rollback, version pinning, and incident review.

## Zug Checks

Run compatibility checks before every deployment:

```powershell
zug check path\to\policy.onnx
zug check path\to\policy.wasm --kind wasm --json
zug check path\to\workload --kind workload --json
```

For agent targets:

```powershell
zug agent --node cell-a --profile vision-f32-basic --listen 127.0.0.1:7070
```

Then query:

```text
GET /health
GET /capabilities
```

The next planned agent step is `POST /workloads/check`, which should return whether a node can accept a robotics policy workload before execution.

Do not add `POST /workloads/run` for robotics policies until the local `zug run workload/` path exists and can enforce action envelopes, resource limits, structured errors, cancellation, and bounded inputs.

## Logging Requirements

Each policy decision should log:

- Workload name and version.
- Guest module hash.
- Model hash.
- Target profile.
- Observation timestamp.
- Input descriptor.
- Output descriptor.
- Action envelope decision.
- Rejection reason when rejected.
- Runtime duration.
- Memory usage where available.
- Node ID.

## Failure Behavior

The default failure behavior must be conservative:

- Invalid input: reject action.
- Invalid output: reject action.
- Shape or dtype mismatch: reject action.
- Missing heartbeat: fallback.
- Stale observation: fallback.
- Runtime error: fallback.
- Policy not allowlisted: reject action.
- Target profile mismatch: reject deployment.

Fallback should be defined by the industrial controller and safety review, not guessed by the model runtime.
