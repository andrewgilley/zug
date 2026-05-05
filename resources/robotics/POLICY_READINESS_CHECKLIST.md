# Robotics Policy Readiness Checklist

Use this checklist before allowing a model-backed WASM workload to influence industrial operations.

## Policy Identity

- [ ] Policy has a stable name.
- [ ] Policy has a version.
- [ ] Model artifact has a hash.
- [ ] WASM guest artifact has a hash.
- [ ] Training data or source policy lineage is recorded.
- [ ] Owner and reviewer are recorded.
- [ ] Intended robot, cell, or process is identified.

## Model And Runtime Compatibility

- [ ] ONNX model runs through `zug inspect`.
- [ ] ONNX model passes `zug check`.
- [ ] WASM guest passes `zug check --kind wasm`.
- [ ] Workload passes `zug check workload/ --kind workload`.
- [ ] Required target profile is explicit.
- [ ] Required memory is explicit.
- [ ] Required dtype is explicit.
- [ ] Network requirements are explicit.
- [ ] Unsupported ops, dtypes, imports, sections, opcodes, and memory features are resolved.

## Observation Contract

- [ ] Input sources are listed.
- [ ] Input shapes are listed.
- [ ] Input dtypes are listed.
- [ ] Coordinate frames are listed.
- [ ] Units are listed.
- [ ] Sensor timestamp requirements are listed.
- [ ] Maximum allowed observation age is defined.
- [ ] Missing or partial observations have defined behavior.

## Action Contract

- [ ] Action vector shape is listed.
- [ ] Action dtype is listed.
- [ ] Action units are listed.
- [ ] Coordinate frame is listed.
- [ ] Action horizon or duration is listed.
- [ ] Joint, velocity, acceleration, force, or process limits are listed.
- [ ] Rate limits are listed.
- [ ] Keep-out zones are listed.
- [ ] Fallback action is listed.

## Safety Boundary

- [ ] Policy output is treated as an action proposal.
- [ ] Host-side action validation exists.
- [ ] Independent safety controls remain active.
- [ ] Emergency stop behavior is documented.
- [ ] Human override behavior is documented.
- [ ] Robot/controller limits are configured outside the model.
- [ ] Safety PLC or safety-rated controller boundary is documented when applicable.

## Rollout Evidence

- [ ] Offline replay completed.
- [ ] Simulation completed.
- [ ] Hardware-in-the-loop completed.
- [ ] Shadow mode completed.
- [ ] Advisory mode completed if required.
- [ ] Supervised bounded actuation completed if required.
- [ ] Production rollout has a rollback plan.

## Cybersecurity

- [ ] Model and WASM artifacts are pinned by hash.
- [ ] Workload source is trusted.
- [ ] Agent endpoint is not exposed without authentication.
- [ ] Network ingress is minimized.
- [ ] Network egress is minimized.
- [ ] Logs avoid leaking sensitive plant or production data.
- [ ] Update and rollback procedures are defined.

## Observability

- [ ] Policy decisions are logged.
- [ ] Action envelope decisions are logged.
- [ ] Rejected actions include reasons.
- [ ] Runtime errors include structured causes.
- [ ] Metrics include latency and memory where available.
- [ ] Model and guest versions appear in logs.
- [ ] Incident review data can be reproduced.

## Go Or No-Go

Do not deploy if any of these are unresolved:

- [ ] Policy can command action outside the validated envelope.
- [ ] Final actuation depends only on model output.
- [ ] Runtime compatibility check fails.
- [ ] Model output shape or dtype is not fixed or validated.
- [ ] Fallback behavior is undefined.
- [ ] Independent safety controls are disabled or bypassed.
- [ ] Agent control endpoint is exposed without security controls.
