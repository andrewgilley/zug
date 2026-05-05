# Robotics Resources

This directory contains project resources for deploying model-backed robotics policies to edge devices and industrial environments.

These resources are not a certification package. They are engineering scaffolding for building a safer runtime workflow around WASM guests, ONNX models, WASI-NN inference, workload manifests, and edge agents.

## Files

- `EDGE_POLICY_DEPLOYMENT.md`: architecture and staged deployment workflow.
- `POLICY_READINESS_CHECKLIST.md`: practical readiness checklist before a policy can influence operations.
- `POLICY_CARD_TEMPLATE.md`: documentation template for a robotics policy.
- `SAFETY_CASE_TEMPLATE.md`: safety case evidence template.
- `policy_manifest_template.toml`: current `zug.toml` template for a robotics policy workload.

## Intended Use

Use these resources when adding or evaluating a policy that can influence robot, machine, or industrial process behavior.

Recommended workflow:

1. Create a workload directory with `policy.wasm`, `policy.onnx`, and `zug.toml`.
2. Fill out a policy card.
3. Run `zug check workload/ --kind workload --json`.
4. Run offline replay and simulation before hardware testing.
5. Keep action validation outside the guest and inside the host/runtime boundary.
6. Keep final actuation behind independent industrial safety controls.

## Alignment With SYSTEMS.md

These resources follow the system architecture in `SYSTEMS.md`:

- Workload-first deployment.
- Capability checks before execution.
- Host-controlled WASI and WASI-NN imports.
- Target profiles for placement.
- Agent capability discovery before remote workload execution.
- Guest action proposals validated by host-side policy before any controller adapter can use them.

## Project Assumption

`zug` is a policy and inference runtime. It is not a safety PLC, robot controller, or certified safety function. It should provide compatibility checks, model execution, bounded host capabilities, logs, and action validation hooks.
