# Safety Case Template

## Scope

- Workload:
- Robot, machine, or process:
- Site or cell:
- Deployment stage:
- Safety reviewer:

## Hazard Summary

List credible hazards related to the policy output.

| Hazard | Cause | Consequence | Mitigation | Evidence |
| --- | --- | --- | --- | --- |
| | | | | |

## Runtime Boundary

- Guest module responsibilities:
- Host runtime responsibilities:
- Controller responsibilities:
- Independent safety system responsibilities:

## Action Envelope

| Constraint | Limit | Source | Validation Method |
| --- | --- | --- | --- |
| Joint position | | | |
| Velocity | | | |
| Acceleration | | | |
| Force or torque | | | |
| Workspace | | | |
| Keep-out zone | | | |
| Observation age | | | |

## Failure Modes

| Failure | Runtime Behavior | Controller Behavior | Evidence |
| --- | --- | --- | --- |
| Invalid input shape | Reject action | | |
| Invalid output shape | Reject action | | |
| Runtime error | Fallback | | |
| Stale observation | Fallback | | |
| Missing heartbeat | Fallback | | |
| Network interruption | Fallback | | |

## Verification Evidence

- `zug check` result:
- Offline replay result:
- Simulation result:
- Hardware-in-the-loop result:
- Shadow mode result:
- Advisory mode result:
- Supervised actuation result:

## Open Risks

| Risk | Owner | Due Date | Decision |
| --- | --- | --- | --- |
| | | | |

## Approval

- Engineering owner:
- Safety owner:
- Operations owner:
- Date:
