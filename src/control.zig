const std = @import("std");

pub const CommandKind = enum {
    check,
    deploy,
    invoke,
    pause,
    @"resume",
    drain,
    remove,
    rollback,
};

pub const DesiredState = enum {
    absent,
    deployed,
    running,
    paused,
};

pub const ObservedState = enum {
    missing,
    accepted,
    loaded,
    running,
    paused,
    failed,
    draining,
    removed,
};

pub const NodeHealth = enum {
    unknown,
    healthy,
    degraded,
    offline,
};

pub const WorkloadRefInput = struct {
    id: []const u8,
    revision: []const u8,
    manifest_hash: []const u8 = "",
    wasm_hash: []const u8 = "",
    model_hash: []const u8 = "",
};

pub const WorkloadRef = struct {
    id: []const u8,
    revision: []const u8,
    manifest_hash: []const u8,
    wasm_hash: []const u8,
    model_hash: []const u8,

    pub fn init(allocator: std.mem.Allocator, input: WorkloadRefInput) !WorkloadRef {
        if (input.id.len == 0) return error.MissingWorkloadId;
        if (input.revision.len == 0) return error.MissingWorkloadRevision;

        const id = try allocator.dupe(u8, input.id);
        errdefer allocator.free(id);

        const revision = try allocator.dupe(u8, input.revision);
        errdefer allocator.free(revision);

        const manifest_hash = try allocator.dupe(u8, input.manifest_hash);
        errdefer allocator.free(manifest_hash);

        const wasm_hash = try allocator.dupe(u8, input.wasm_hash);
        errdefer allocator.free(wasm_hash);

        const model_hash = try allocator.dupe(u8, input.model_hash);
        errdefer allocator.free(model_hash);

        return .{
            .id = id,
            .revision = revision,
            .manifest_hash = manifest_hash,
            .wasm_hash = wasm_hash,
            .model_hash = model_hash,
        };
    }

    pub fn clone(self: WorkloadRef, allocator: std.mem.Allocator) !WorkloadRef {
        return init(allocator, .{
            .id = self.id,
            .revision = self.revision,
            .manifest_hash = self.manifest_hash,
            .wasm_hash = self.wasm_hash,
            .model_hash = self.model_hash,
        });
    }

    pub fn deinit(self: *WorkloadRef, allocator: std.mem.Allocator) void {
        allocator.free(self.id);
        allocator.free(self.revision);
        allocator.free(self.manifest_hash);
        allocator.free(self.wasm_hash);
        allocator.free(self.model_hash);
        self.* = undefined;
    }

    pub fn sameRevision(self: WorkloadRef, other: WorkloadRef) bool {
        return std.mem.eql(u8, self.id, other.id) and
            std.mem.eql(u8, self.revision, other.revision);
    }
};

pub const DeploymentIntentInput = struct {
    node_id: []const u8,
    workload: WorkloadRefInput,
    desired_state: DesiredState,
    generation: u64 = 1,
};

pub const DeploymentIntent = struct {
    node_id: []const u8,
    workload: WorkloadRef,
    desired_state: DesiredState,
    generation: u64,

    pub fn init(allocator: std.mem.Allocator, input: DeploymentIntentInput) !DeploymentIntent {
        if (input.node_id.len == 0) return error.MissingNodeId;

        const node_id = try allocator.dupe(u8, input.node_id);
        errdefer allocator.free(node_id);

        var workload = try WorkloadRef.init(allocator, input.workload);
        errdefer workload.deinit(allocator);

        return .{
            .node_id = node_id,
            .workload = workload,
            .desired_state = input.desired_state,
            .generation = input.generation,
        };
    }

    pub fn deinit(self: *DeploymentIntent, allocator: std.mem.Allocator) void {
        allocator.free(self.node_id);
        self.workload.deinit(allocator);
        self.* = undefined;
    }
};

pub const NodeObservationInput = struct {
    node_id: []const u8,
    profile: []const u8,
    health: NodeHealth = .unknown,
    last_seen_ns: u64 = 0,
};

pub const NodeObservation = struct {
    node_id: []const u8,
    profile: []const u8,
    health: NodeHealth,
    last_seen_ns: u64,

    pub fn init(allocator: std.mem.Allocator, input: NodeObservationInput) !NodeObservation {
        if (input.node_id.len == 0) return error.MissingNodeId;

        const node_id = try allocator.dupe(u8, input.node_id);
        errdefer allocator.free(node_id);

        const profile = try allocator.dupe(u8, input.profile);
        errdefer allocator.free(profile);

        return .{
            .node_id = node_id,
            .profile = profile,
            .health = input.health,
            .last_seen_ns = input.last_seen_ns,
        };
    }

    pub fn deinit(self: *NodeObservation, allocator: std.mem.Allocator) void {
        allocator.free(self.node_id);
        allocator.free(self.profile);
        self.* = undefined;
    }

    pub fn canAcceptControl(self: NodeObservation) bool {
        return self.health == .healthy or self.health == .degraded;
    }
};

pub const DeploymentObservationInput = struct {
    node_id: []const u8,
    workload: WorkloadRefInput,
    observed_state: ObservedState,
    generation: u64 = 0,
    last_activity_id: u64 = 0,
    last_error: []const u8 = "",
    observed_at_ns: u64 = 0,
};

pub const DeploymentObservation = struct {
    node_id: []const u8,
    workload: WorkloadRef,
    observed_state: ObservedState,
    generation: u64,
    last_activity_id: u64,
    last_error: []const u8,
    observed_at_ns: u64,

    pub fn init(allocator: std.mem.Allocator, input: DeploymentObservationInput) !DeploymentObservation {
        if (input.node_id.len == 0) return error.MissingNodeId;

        const node_id = try allocator.dupe(u8, input.node_id);
        errdefer allocator.free(node_id);

        var workload = try WorkloadRef.init(allocator, input.workload);
        errdefer workload.deinit(allocator);

        const last_error = try allocator.dupe(u8, input.last_error);
        errdefer allocator.free(last_error);

        return .{
            .node_id = node_id,
            .workload = workload,
            .observed_state = input.observed_state,
            .generation = input.generation,
            .last_activity_id = input.last_activity_id,
            .last_error = last_error,
            .observed_at_ns = input.observed_at_ns,
        };
    }

    pub fn deinit(self: *DeploymentObservation, allocator: std.mem.Allocator) void {
        allocator.free(self.node_id);
        self.workload.deinit(allocator);
        allocator.free(self.last_error);
        self.* = undefined;
    }

    pub fn belongsTo(self: DeploymentObservation, intent: DeploymentIntent) bool {
        return std.mem.eql(u8, self.node_id, intent.node_id) and
            std.mem.eql(u8, self.workload.id, intent.workload.id);
    }
};

pub const Command = struct {
    id: u64,
    kind: CommandKind,
    node_id: []const u8,
    workload: WorkloadRef,
    generation: u64,
    reason: []const u8,

    pub fn init(
        allocator: std.mem.Allocator,
        id: u64,
        kind: CommandKind,
        intent: DeploymentIntent,
        reason: []const u8,
    ) !Command {
        const node_id = try allocator.dupe(u8, intent.node_id);
        errdefer allocator.free(node_id);

        var workload = try intent.workload.clone(allocator);
        errdefer workload.deinit(allocator);

        const owned_reason = try allocator.dupe(u8, reason);
        errdefer allocator.free(owned_reason);

        return .{
            .id = id,
            .kind = kind,
            .node_id = node_id,
            .workload = workload,
            .generation = intent.generation,
            .reason = owned_reason,
        };
    }

    pub fn deinit(self: *Command, allocator: std.mem.Allocator) void {
        allocator.free(self.node_id);
        self.workload.deinit(allocator);
        allocator.free(self.reason);
        self.* = undefined;
    }
};

pub const Plan = struct {
    next_command_id: u64 = 1,
    commands: std.ArrayList(Command) = .empty,

    pub fn deinit(self: *Plan, allocator: std.mem.Allocator) void {
        for (self.commands.items) |*command| {
            command.deinit(allocator);
        }
        self.commands.deinit(allocator);
        self.* = undefined;
    }

    pub fn append(
        self: *Plan,
        allocator: std.mem.Allocator,
        kind: CommandKind,
        intent: DeploymentIntent,
        reason: []const u8,
    ) !void {
        const command = try Command.init(allocator, self.next_command_id, kind, intent, reason);
        errdefer {
            var owned = command;
            owned.deinit(allocator);
        }

        self.next_command_id += 1;
        try self.commands.append(allocator, command);
    }
};

pub fn reconcileOne(
    allocator: std.mem.Allocator,
    plan: *Plan,
    intent: DeploymentIntent,
    observation: ?DeploymentObservation,
) !void {
    if (observation) |observed| {
        if (!observed.belongsTo(intent)) return error.ControlObservationMismatch;
    }

    switch (intent.desired_state) {
        .absent => {
            if (observation) |observed| {
                if (!isAbsent(observed.observed_state)) {
                    try plan.append(allocator, .remove, intent, "desired_absent");
                }
            }
        },
        .deployed => {
            if (observation) |observed| {
                if (!observed.workload.sameRevision(intent.workload)) {
                    try plan.append(allocator, .deploy, intent, "revision_mismatch");
                } else if (!isDeployed(observed.observed_state)) {
                    try plan.append(allocator, .deploy, intent, "not_deployed");
                }
            } else {
                try plan.append(allocator, .deploy, intent, "missing_observation");
            }
        },
        .running => {
            if (observation) |observed| {
                if (!observed.workload.sameRevision(intent.workload)) {
                    try plan.append(allocator, .deploy, intent, "revision_mismatch");
                } else switch (observed.observed_state) {
                    .running => {},
                    .paused => try plan.append(allocator, .@"resume", intent, "paused"),
                    .accepted, .loaded => try plan.append(allocator, .invoke, intent, "deployed_not_running"),
                    else => try plan.append(allocator, .deploy, intent, "not_deployed"),
                }
            } else {
                try plan.append(allocator, .deploy, intent, "missing_observation");
            }
        },
        .paused => {
            if (observation) |observed| {
                if (!observed.workload.sameRevision(intent.workload)) {
                    try plan.append(allocator, .deploy, intent, "revision_mismatch");
                } else switch (observed.observed_state) {
                    .paused => {},
                    .running => try plan.append(allocator, .pause, intent, "running"),
                    .accepted, .loaded => {},
                    else => try plan.append(allocator, .deploy, intent, "not_deployed"),
                }
            } else {
                try plan.append(allocator, .deploy, intent, "missing_observation");
            }
        },
    }
}

pub fn reconcile(
    allocator: std.mem.Allocator,
    intents: []const DeploymentIntent,
    observations: []const DeploymentObservation,
) !Plan {
    var plan = Plan{};
    errdefer plan.deinit(allocator);

    for (intents) |intent| {
        try reconcileOne(allocator, &plan, intent, findObservation(intent, observations));
    }

    return plan;
}

pub fn commandKindName(kind: CommandKind) []const u8 {
    return switch (kind) {
        .check => "check",
        .deploy => "deploy",
        .invoke => "invoke",
        .pause => "pause",
        .@"resume" => "resume",
        .drain => "drain",
        .remove => "remove",
        .rollback => "rollback",
    };
}

pub fn desiredStateName(state: DesiredState) []const u8 {
    return switch (state) {
        .absent => "absent",
        .deployed => "deployed",
        .running => "running",
        .paused => "paused",
    };
}

pub fn observedStateName(state: ObservedState) []const u8 {
    return switch (state) {
        .missing => "missing",
        .accepted => "accepted",
        .loaded => "loaded",
        .running => "running",
        .paused => "paused",
        .failed => "failed",
        .draining => "draining",
        .removed => "removed",
    };
}

pub fn nodeHealthName(health: NodeHealth) []const u8 {
    return switch (health) {
        .unknown => "unknown",
        .healthy => "healthy",
        .degraded => "degraded",
        .offline => "offline",
    };
}

fn findObservation(
    intent: DeploymentIntent,
    observations: []const DeploymentObservation,
) ?DeploymentObservation {
    for (observations) |observation| {
        if (observation.belongsTo(intent)) return observation;
    }

    return null;
}

fn isAbsent(state: ObservedState) bool {
    return state == .missing or state == .removed;
}

fn isDeployed(state: ObservedState) bool {
    return switch (state) {
        .accepted, .loaded, .running, .paused => true,
        else => false,
    };
}

test "control reconciler deploys missing workload intent" {
    const allocator = std.testing.allocator;

    var intent = try DeploymentIntent.init(allocator, .{
        .node_id = "edge-a",
        .workload = .{
            .id = "camera-classifier",
            .revision = "rev-001",
            .manifest_hash = "sha256:manifest",
            .wasm_hash = "sha256:wasm",
            .model_hash = "sha256:model",
        },
        .desired_state = .deployed,
        .generation = 7,
    });
    defer intent.deinit(allocator);

    var plan = try reconcile(allocator, &.{intent}, &.{});
    defer plan.deinit(allocator);

    try std.testing.expectEqual(@as(usize, 1), plan.commands.items.len);
    try std.testing.expectEqual(CommandKind.deploy, plan.commands.items[0].kind);
    try std.testing.expectEqual(@as(u64, 1), plan.commands.items[0].id);
    try std.testing.expectEqual(@as(u64, 7), plan.commands.items[0].generation);
    try std.testing.expectEqualStrings("edge-a", plan.commands.items[0].node_id);
    try std.testing.expectEqualStrings("camera-classifier", plan.commands.items[0].workload.id);
    try std.testing.expectEqualStrings("missing_observation", plan.commands.items[0].reason);
}

test "control reconciler emits explicit lifecycle commands" {
    const allocator = std.testing.allocator;

    var running_intent = try DeploymentIntent.init(allocator, .{
        .node_id = "edge-a",
        .workload = .{ .id = "policy", .revision = "rev-002" },
        .desired_state = .running,
    });
    defer running_intent.deinit(allocator);

    var paused_observation = try DeploymentObservation.init(allocator, .{
        .node_id = "edge-a",
        .workload = .{ .id = "policy", .revision = "rev-002" },
        .observed_state = .paused,
    });
    defer paused_observation.deinit(allocator);

    var absent_intent = try DeploymentIntent.init(allocator, .{
        .node_id = "edge-b",
        .workload = .{ .id = "policy", .revision = "rev-002" },
        .desired_state = .absent,
    });
    defer absent_intent.deinit(allocator);

    var loaded_observation = try DeploymentObservation.init(allocator, .{
        .node_id = "edge-b",
        .workload = .{ .id = "policy", .revision = "rev-002" },
        .observed_state = .loaded,
    });
    defer loaded_observation.deinit(allocator);

    var plan = try reconcile(
        allocator,
        &.{ running_intent, absent_intent },
        &.{ paused_observation, loaded_observation },
    );
    defer plan.deinit(allocator);

    try std.testing.expectEqual(@as(usize, 2), plan.commands.items.len);
    try std.testing.expectEqual(CommandKind.@"resume", plan.commands.items[0].kind);
    try std.testing.expectEqualStrings("paused", plan.commands.items[0].reason);
    try std.testing.expectEqual(CommandKind.remove, plan.commands.items[1].kind);
    try std.testing.expectEqualStrings("desired_absent", plan.commands.items[1].reason);
}

test "control reconciler treats matching deployed revision as converged" {
    const allocator = std.testing.allocator;

    var intent = try DeploymentIntent.init(allocator, .{
        .node_id = "edge-a",
        .workload = .{ .id = "classifier", .revision = "rev-003" },
        .desired_state = .deployed,
    });
    defer intent.deinit(allocator);

    var observation = try DeploymentObservation.init(allocator, .{
        .node_id = "edge-a",
        .workload = .{ .id = "classifier", .revision = "rev-003" },
        .observed_state = .loaded,
    });
    defer observation.deinit(allocator);

    var plan = try reconcile(allocator, &.{intent}, &.{observation});
    defer plan.deinit(allocator);

    try std.testing.expectEqual(@as(usize, 0), plan.commands.items.len);
}

test "control names expose stable protocol strings" {
    try std.testing.expectEqualStrings("deploy", commandKindName(.deploy));
    try std.testing.expectEqualStrings("running", desiredStateName(.running));
    try std.testing.expectEqualStrings("failed", observedStateName(.failed));
    try std.testing.expectEqualStrings("healthy", nodeHealthName(.healthy));
}
