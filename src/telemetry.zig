const std = @import("std");

pub const Severity = enum {
    info,
    warn,
    err,
};

pub const EventInput = struct {
    source: []const u8 = "agent",
    kind: []const u8,
    message: []const u8,
    severity: Severity = .info,
};

pub const MetricInput = struct {
    source: []const u8 = "agent",
    name: []const u8,
    value: f64,
    unit: []const u8 = "",
};

pub const SpanKind = enum {
    unspecified,
    internal,
    server,
    client,
    producer,
    consumer,
};

pub const SpanStatus = enum {
    unset,
    ok,
    err,
};

pub const TraceInput = struct {
    source: []const u8 = "agent",
    trace_id: []const u8,
    span_id: []const u8,
    parent_span_id: []const u8 = "",
    name: []const u8,
    kind: SpanKind = .internal,
    start_time_ns: u64,
    end_time_ns: u64,
    status: SpanStatus = .unset,
};

pub const Event = struct {
    id: u64,
    timestamp_ns: u64,
    source: []const u8,
    kind: []const u8,
    message: []const u8,
    severity: Severity,

    fn deinit(self: *Event, allocator: std.mem.Allocator) void {
        allocator.free(self.source);
        allocator.free(self.kind);
        allocator.free(self.message);
        self.* = undefined;
    }
};

pub const Metric = struct {
    id: u64,
    timestamp_ns: u64,
    source: []const u8,
    name: []const u8,
    value: f64,
    unit: []const u8,

    fn deinit(self: *Metric, allocator: std.mem.Allocator) void {
        allocator.free(self.source);
        allocator.free(self.name);
        allocator.free(self.unit);
        self.* = undefined;
    }
};

pub const TraceSpan = struct {
    id: u64,
    source: []const u8,
    trace_id: []const u8,
    span_id: []const u8,
    parent_span_id: []const u8,
    name: []const u8,
    kind: SpanKind,
    start_time_ns: u64,
    end_time_ns: u64,
    status: SpanStatus,

    fn deinit(self: *TraceSpan, allocator: std.mem.Allocator) void {
        allocator.free(self.source);
        allocator.free(self.trace_id);
        allocator.free(self.span_id);
        allocator.free(self.parent_span_id);
        allocator.free(self.name);
        self.* = undefined;
    }
};

pub const Store = struct {
    next_event_id: u64 = 1,
    next_metric_id: u64 = 1,
    next_trace_id: u64 = 1,
    events: std.ArrayList(Event) = .empty,
    metrics: std.ArrayList(Metric) = .empty,
    traces: std.ArrayList(TraceSpan) = .empty,

    pub fn deinit(self: *Store, allocator: std.mem.Allocator) void {
        for (self.events.items) |*event| {
            event.deinit(allocator);
        }
        self.events.deinit(allocator);

        for (self.metrics.items) |*metric| {
            metric.deinit(allocator);
        }
        self.metrics.deinit(allocator);

        for (self.traces.items) |*span| {
            span.deinit(allocator);
        }
        self.traces.deinit(allocator);

        self.* = undefined;
    }

    pub fn recordEvent(
        self: *Store,
        allocator: std.mem.Allocator,
        input: EventInput,
        timestamp_ns: u64,
    ) !Event {
        if (input.kind.len == 0) return error.MissingTelemetryKind;
        if (input.message.len == 0) return error.MissingTelemetryMessage;

        const source = try allocator.dupe(u8, input.source);
        errdefer allocator.free(source);

        const kind = try allocator.dupe(u8, input.kind);
        errdefer allocator.free(kind);

        const message = try allocator.dupe(u8, input.message);
        errdefer allocator.free(message);

        const event = Event{
            .id = self.next_event_id,
            .timestamp_ns = timestamp_ns,
            .source = source,
            .kind = kind,
            .message = message,
            .severity = input.severity,
        };
        self.next_event_id += 1;

        try self.events.append(allocator, event);
        return event;
    }

    pub fn recordMetric(
        self: *Store,
        allocator: std.mem.Allocator,
        input: MetricInput,
        timestamp_ns: u64,
    ) !Metric {
        if (input.name.len == 0) return error.MissingTelemetryMetricName;
        if (!std.math.isFinite(input.value)) return error.InvalidTelemetryMetricValue;

        const source = try allocator.dupe(u8, input.source);
        errdefer allocator.free(source);

        const name = try allocator.dupe(u8, input.name);
        errdefer allocator.free(name);

        const unit = try allocator.dupe(u8, input.unit);
        errdefer allocator.free(unit);

        const metric = Metric{
            .id = self.next_metric_id,
            .timestamp_ns = timestamp_ns,
            .source = source,
            .name = name,
            .value = input.value,
            .unit = unit,
        };
        self.next_metric_id += 1;

        try self.metrics.append(allocator, metric);
        return metric;
    }

    pub fn recordTrace(
        self: *Store,
        allocator: std.mem.Allocator,
        input: TraceInput,
    ) !TraceSpan {
        if (input.trace_id.len == 0) return error.MissingTelemetryTraceId;
        if (input.span_id.len == 0) return error.MissingTelemetrySpanId;
        if (input.name.len == 0) return error.MissingTelemetrySpanName;
        if (input.end_time_ns != 0 and input.end_time_ns < input.start_time_ns) {
            return error.InvalidTelemetrySpanTimeRange;
        }

        const source = try allocator.dupe(u8, input.source);
        errdefer allocator.free(source);

        const trace_id = try allocator.dupe(u8, input.trace_id);
        errdefer allocator.free(trace_id);

        const span_id = try allocator.dupe(u8, input.span_id);
        errdefer allocator.free(span_id);

        const parent_span_id = try allocator.dupe(u8, input.parent_span_id);
        errdefer allocator.free(parent_span_id);

        const name = try allocator.dupe(u8, input.name);
        errdefer allocator.free(name);

        const span = TraceSpan{
            .id = self.next_trace_id,
            .source = source,
            .trace_id = trace_id,
            .span_id = span_id,
            .parent_span_id = parent_span_id,
            .name = name,
            .kind = input.kind,
            .start_time_ns = input.start_time_ns,
            .end_time_ns = input.end_time_ns,
            .status = input.status,
        };
        self.next_trace_id += 1;

        try self.traces.append(allocator, span);
        return span;
    }
};

pub fn severityName(severity: Severity) []const u8 {
    return switch (severity) {
        .info => "info",
        .warn => "warn",
        .err => "error",
    };
}

pub fn parseSeverity(value: []const u8) ?Severity {
    if (std.mem.eql(u8, value, "info")) return .info;
    if (std.mem.eql(u8, value, "warn")) return .warn;
    if (std.mem.eql(u8, value, "warning")) return .warn;
    if (std.mem.eql(u8, value, "error")) return .err;
    if (std.mem.eql(u8, value, "err")) return .err;
    return null;
}

pub fn spanKindName(kind: SpanKind) []const u8 {
    return switch (kind) {
        .unspecified => "unspecified",
        .internal => "internal",
        .server => "server",
        .client => "client",
        .producer => "producer",
        .consumer => "consumer",
    };
}

pub fn parseSpanKind(value: []const u8) ?SpanKind {
    if (std.mem.eql(u8, value, "unspecified")) return .unspecified;
    if (std.mem.eql(u8, value, "internal")) return .internal;
    if (std.mem.eql(u8, value, "server")) return .server;
    if (std.mem.eql(u8, value, "client")) return .client;
    if (std.mem.eql(u8, value, "producer")) return .producer;
    if (std.mem.eql(u8, value, "consumer")) return .consumer;
    return null;
}

pub fn spanStatusName(status: SpanStatus) []const u8 {
    return switch (status) {
        .unset => "unset",
        .ok => "ok",
        .err => "error",
    };
}

pub fn parseSpanStatus(value: []const u8) ?SpanStatus {
    if (std.mem.eql(u8, value, "unset")) return .unset;
    if (std.mem.eql(u8, value, "ok")) return .ok;
    if (std.mem.eql(u8, value, "error")) return .err;
    if (std.mem.eql(u8, value, "err")) return .err;
    return null;
}

test "telemetry store records events metrics and traces" {
    var store = Store{};
    defer store.deinit(std.testing.allocator);

    const event = try store.recordEvent(std.testing.allocator, .{
        .source = "camera",
        .kind = "frame_drop",
        .message = "dropped frame",
        .severity = .warn,
    }, 11);

    const metric = try store.recordMetric(std.testing.allocator, .{
        .source = "runtime",
        .name = "latency_ms",
        .value = 14.5,
        .unit = "ms",
    }, 12);

    const span = try store.recordTrace(std.testing.allocator, .{
        .source = "runtime",
        .trace_id = "5b8efff798038103d269b633813fc60c",
        .span_id = "eee19b7ec3c1b174",
        .name = "invoke",
        .kind = .server,
        .start_time_ns = 13,
        .end_time_ns = 17,
        .status = .ok,
    });

    try std.testing.expectEqual(@as(u64, 1), event.id);
    try std.testing.expectEqual(@as(u64, 1), metric.id);
    try std.testing.expectEqual(@as(u64, 1), span.id);
    try std.testing.expectEqual(@as(usize, 1), store.events.items.len);
    try std.testing.expectEqual(@as(usize, 1), store.metrics.items.len);
    try std.testing.expectEqual(@as(usize, 1), store.traces.items.len);
    try std.testing.expectEqualStrings("warn", severityName(store.events.items[0].severity));
    try std.testing.expectEqualStrings("server", spanKindName(store.traces.items[0].kind));
}
