# Robotics Policy Deployment For Edge Operations

## Purpose

`zug` is moving toward edge deployments where ML models and WASM guest logic can inform industrial actions. Robotics policies are one of the strongest examples: a guest module can collect observations, call a model through WASI-NN, postprocess outputs, and produce an action proposal close to a machine, robot cell, or industrial process.

This document defines how robotics policy deployment should fit into the project.

## Safety Boundary

`zug` should not be treated as a safety-rated controller.

The runtime can execute policy logic, run model inference, check compatibility, report telemetry, and produce action proposals. Final actuation in an industrial environment must remain behind independent safety controls, machine interlocks, emergency stops, safety PLCs, robot controller limits, and site-specific risk reduction measures.

The runtime should be designed so policy output is always bounded by a host-side action envelope before it can affect equipment.

Required boundary:

```text
sensor input -> WASM policy guest -> WASI-NN model inference -> action proposal
    -> host action validator -> robot/controller interface -> independent safety layer
```

The host action validator is part of the `zug` direction. The independent safety layer is outside `zug` and must not be bypassed.

## Current Project Fit

Existing `zug` capabilities already support the first version of this workflow:

- ONNX policy or perception model loading.
- WASM guest logic for preprocessing, postprocessing, and policy glue.
- WASI-NN shaped graph execution.
- Workload manifests through `zug.toml`.
- Target profiles and memory/model limits.
- Network requirement checks.
- `zug check` for model, WASM, and workload compatibility.
- `zug agent` for node capability reporting over HTTP.

The missing robotics-specific pieces are action validation, policy metadata, rollout evidence, hardware interface adapters, and deployment gates.

## Alignment With SYSTEMS.md

Robotics resources should follow the same architecture defined in `SYSTEMS.md`:

- Treat a robotics policy as a `zug` workload: guest WASM, model artifact, and `zug.toml`.
- Keep guest code portable and sandboxed.
- Put model execution behind the WASI-NN shaped host ABI.
- Keep host authority in the runtime: compatibility checks, resource limits, network policy, and action validation.
- Use target profiles to describe edge node capability.
- Use `zug check workload/ --kind workload` before any execution.
- Use `zug agent` first for capability discovery, not remote actuation.
- Add local `zug run workload/` before adding remote `POST /workloads/run`.
- Keep guest networking separate from host control-plane networking.
- Treat every model output as an action proposal until the host action envelope accepts it.

## Robotics Workload Shape

A robotics policy workload should eventually look like this:

```text
robot-policy/
  policy.wasm
  policy.onnx
  zug.toml
  POLICY_CARD.md
  SAFETY_CASE.md
  calibration/
  tests/
```

Current `zug.toml` can describe the runtime requirements. The additional policy and safety files provide human-readable evidence until the runtime has first-class schema support for robotics metadata.

## Policy Runtime Contract

Robotics policy execution should produce an action proposal, not unchecked actuation.

Minimum action proposal fields:

- Policy version.
- Observation timestamp.
- Input sensor frame IDs or source IDs.
- Action vector or command payload.
- Action units and coordinate frame.
- Confidence or validity status when available.
- Model output checksum or trace ID.
- Proposed duration or time horizon.
- Host validation result.

Minimum host-side validation:

- Joint, velocity, acceleration, and force limits.
- Workspace and keep-out-zone checks.
- Rate-of-change limits.
- Deadman or heartbeat requirement.
- Observation freshness check.
- Policy version allowlist.
- Input shape and dtype check.
- Output shape and dtype check.
- Fallback action on invalid output.

## Deployment Stages

Robotics policies should move through explicit rollout gates:

1. Offline replay against recorded sensor and state data.
2. Simulation with deterministic seeds and logged actions.
3. Hardware-in-the-loop without physical actuation.
4. Shadow mode beside the production controller.
5. Advisory mode with human or PLC approval.
6. Supervised actuation inside reduced speed and force limits.
7. Bounded production deployment with rollback.

No robotics policy should move directly from model execution to production actuation.

## Resources Added

Robotics deployment resources live under `resources/robotics/`:

- `README.md`: resource index.
- `EDGE_POLICY_DEPLOYMENT.md`: deployment architecture and staged workflow.
- `POLICY_READINESS_CHECKLIST.md`: deployment checklist.
- `POLICY_CARD_TEMPLATE.md`: model and policy documentation template.
- `policy_manifest_template.toml`: current-schema workload manifest template.

## Standards And Risk References

These references should guide product decisions, but they do not make `zug` compliant by themselves:

- ISO 10218-1:2025 for industrial robot safety: https://www.iso.org/standard/73933.html
- ISO 10218-2:2025 for industrial robot applications and robot cells: https://www.iso.org/standard/73934.html
- ISO 13849-1:2023 for safety-related parts of machinery control systems: https://www.iso.org/standard/73481.html
- IEC 61508 for functional safety of electrical, electronic, and programmable electronic safety-related systems: https://webstore.iec.ch/en/publication/5515
- ISA/IEC 62443 for industrial automation and control systems cybersecurity: https://www.isa.org/standards-and-publications/isa-standards/isa-iec-62443-series-of-standards
- NIST AI RMF 1.0 for AI risk management: https://www.nist.gov/itl/ai-risk-management-framework
- NIST Cybersecurity Framework 2.0 for cybersecurity risk management: https://www.nist.gov/publications/nist-cybersecurity-framework-csf-20

## Project Direction

The robotics path should focus on earning this first product claim:

```text
zug can check, host, and execute a model-backed WASM robotics policy at the edge,
while exposing enough metadata and guardrails for industrial deployment review.
```

The implementation order should stay aligned with `SYSTEMS.md`:

1. Add local workload execution with `zug run workload/`.
2. Add robotics policy metadata and action envelope validation.
3. Add `POST /workloads/check` to the agent.
4. Add remote execution only after local workload execution, limits, request IDs, structured errors, and cancellation exist.
