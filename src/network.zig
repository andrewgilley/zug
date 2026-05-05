const std = @import("std");

pub const Protocol = enum {
    local,
    stdio,
    http,
    https,
    tcp,
    udp,
    nats,
    mqtt,
};

pub const Requirements = struct {
    input: ?Protocol = null,
    output: ?Protocol = null,
    ingress: std.ArrayList(Protocol) = .empty,
    egress: std.ArrayList(Protocol) = .empty,
    max_inflight: ?usize = null,
    request_timeout_ms: ?u64 = null,
    public_egress: bool = false,

    pub fn deinit(self: *Requirements, allocator: std.mem.Allocator) void {
        self.ingress.deinit(allocator);
        self.egress.deinit(allocator);
        self.* = undefined;
    }

    pub fn validate(self: Requirements, caps: Capabilities) !void {
        if (self.input) |protocol| {
            if (!caps.supportsProtocol(protocol)) return error.WorkloadNetworkInputUnsupported;
        }
        if (self.output) |protocol| {
            if (!caps.supportsProtocol(protocol)) return error.WorkloadNetworkOutputUnsupported;
        }

        for (self.ingress.items) |protocol| {
            if (!caps.supportsProtocol(protocol)) return error.WorkloadNetworkIngressUnsupported;
        }
        for (self.egress.items) |protocol| {
            if (!caps.supportsProtocol(protocol)) return error.WorkloadNetworkEgressUnsupported;
        }

        if (self.max_inflight) |requested| {
            if (requested > caps.max_inflight) return error.WorkloadNetworkConcurrencyTooHigh;
        }
        if (self.request_timeout_ms) |requested| {
            if (caps.max_request_timeout_ms != 0 and requested > caps.max_request_timeout_ms) {
                return error.WorkloadNetworkTimeoutTooHigh;
            }
        }
        if (self.public_egress and !caps.public_egress) return error.WorkloadPublicEgressUnsupported;
    }
};

pub const Capabilities = struct {
    protocols: []const Protocol = &.{},
    max_inflight: usize = 1,
    max_request_timeout_ms: u64 = 0,
    public_egress: bool = false,

    pub fn supportsProtocol(self: Capabilities, protocol: Protocol) bool {
        for (self.protocols) |supported| {
            if (supported == protocol) return true;
        }

        return false;
    }
};

pub const ProtocolCheck = struct {
    subject: []const u8,
    protocol: Protocol,
    supported: bool,
};

pub const Report = struct {
    protocol_checks: std.ArrayList(ProtocolCheck) = .empty,
    requested_max_inflight: ?usize = null,
    target_max_inflight: usize = 0,
    max_inflight_supported: bool = true,
    requested_timeout_ms: ?u64 = null,
    target_max_timeout_ms: u64 = 0,
    timeout_supported: bool = true,
    public_egress_requested: bool = false,
    public_egress_supported: bool = true,
    unsupported_count: usize = 0,

    pub fn deinit(self: *Report, allocator: std.mem.Allocator) void {
        self.protocol_checks.deinit(allocator);
        self.* = undefined;
    }

    pub fn supported(self: Report) bool {
        return self.unsupported_count == 0 and
            self.max_inflight_supported and
            self.timeout_supported and
            self.public_egress_supported;
    }
};

pub fn analyze(allocator: std.mem.Allocator, requirements: Requirements, caps: Capabilities) !Report {
    var report = Report{
        .requested_max_inflight = requirements.max_inflight,
        .target_max_inflight = caps.max_inflight,
        .requested_timeout_ms = requirements.request_timeout_ms,
        .target_max_timeout_ms = caps.max_request_timeout_ms,
        .public_egress_requested = requirements.public_egress,
        .public_egress_supported = !requirements.public_egress or caps.public_egress,
    };
    errdefer report.deinit(allocator);

    if (requirements.input) |protocol| {
        try appendProtocolCheck(allocator, &report, "input", protocol, caps.supportsProtocol(protocol));
    }
    if (requirements.output) |protocol| {
        try appendProtocolCheck(allocator, &report, "output", protocol, caps.supportsProtocol(protocol));
    }

    for (requirements.ingress.items) |protocol| {
        try appendProtocolCheck(allocator, &report, "ingress", protocol, caps.supportsProtocol(protocol));
    }
    for (requirements.egress.items) |protocol| {
        try appendProtocolCheck(allocator, &report, "egress", protocol, caps.supportsProtocol(protocol));
    }

    if (requirements.max_inflight) |requested| {
        report.max_inflight_supported = requested <= caps.max_inflight;
        if (!report.max_inflight_supported) report.unsupported_count += 1;
    }

    if (requirements.request_timeout_ms) |requested| {
        report.timeout_supported = caps.max_request_timeout_ms == 0 or requested <= caps.max_request_timeout_ms;
        if (!report.timeout_supported) report.unsupported_count += 1;
    }

    if (!report.public_egress_supported) report.unsupported_count += 1;

    return report;
}

fn appendProtocolCheck(
    allocator: std.mem.Allocator,
    report: *Report,
    subject: []const u8,
    protocol: Protocol,
    supported: bool,
) !void {
    try report.protocol_checks.append(allocator, .{
        .subject = subject,
        .protocol = protocol,
        .supported = supported,
    });

    if (!supported) report.unsupported_count += 1;
}

pub fn parseProtocol(value: []const u8) !Protocol {
    if (std.ascii.eqlIgnoreCase(value, "local")) return .local;
    if (std.ascii.eqlIgnoreCase(value, "stdio") or
        std.ascii.eqlIgnoreCase(value, "stdin") or
        std.ascii.eqlIgnoreCase(value, "stdout") or
        std.ascii.eqlIgnoreCase(value, "stderr"))
    {
        return .stdio;
    }
    if (std.ascii.eqlIgnoreCase(value, "http")) return .http;
    if (std.ascii.eqlIgnoreCase(value, "https")) return .https;
    if (std.ascii.eqlIgnoreCase(value, "tcp")) return .tcp;
    if (std.ascii.eqlIgnoreCase(value, "udp")) return .udp;
    if (std.ascii.eqlIgnoreCase(value, "nats")) return .nats;
    if (std.ascii.eqlIgnoreCase(value, "mqtt")) return .mqtt;

    return error.UnknownNetworkProtocol;
}

pub fn protocolName(protocol: Protocol) []const u8 {
    return switch (protocol) {
        .local => "local",
        .stdio => "stdio",
        .http => "http",
        .https => "https",
        .tcp => "tcp",
        .udp => "udp",
        .nats => "nats",
        .mqtt => "mqtt",
    };
}

test "network report detects unsupported protocols and policy limits" {
    var requirements = Requirements{};
    defer requirements.deinit(std.testing.allocator);

    requirements.input = .http;
    try requirements.egress.append(std.testing.allocator, .nats);
    requirements.max_inflight = 8;
    requirements.request_timeout_ms = 10_000;
    requirements.public_egress = true;

    var report = try analyze(std.testing.allocator, requirements, .{
        .protocols = &.{ .local, .http },
        .max_inflight = 4,
        .max_request_timeout_ms = 5_000,
        .public_egress = false,
    });
    defer report.deinit(std.testing.allocator);

    try std.testing.expect(!report.supported());
    try std.testing.expectEqual(@as(usize, 4), report.unsupported_count);
}

test "network protocol parser accepts stdio aliases" {
    try std.testing.expectEqual(Protocol.stdio, try parseProtocol("stdout"));
    try std.testing.expectEqual(Protocol.https, try parseProtocol("HTTPS"));
    try std.testing.expectError(error.UnknownNetworkProtocol, parseProtocol("bluetooth"));
}
