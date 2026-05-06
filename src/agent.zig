const std = @import("std");
const accelerator = @import("accelerator.zig");
const capabilities = @import("capabilities.zig");
const gpu = @import("gpu.zig");
const network = @import("network.zig");
const onnx = @import("proto/onnx.pb.zig");
const scope = @import("scope.zig");
const target_profile = @import("target.zig");
const telemetry = @import("telemetry.zig");
const wasi_nn_abi = @import("wasi_nn_abi.zig");
const workload = @import("workload.zig");
const wasm_compatibility = @import("wasm/compatibility.zig");
const wasm_imports = @import("wasm/imports.zig");
const wasm_interpreter = @import("wasm/interpreter.zig");
const wasm_runtime = @import("wasm/runtime.zig");

const net = std.Io.net;

pub const Config = struct {
    node_id: []const u8 = "local-node",
    profile_name: []const u8 = "vision-f32-basic",
    listen: []const u8 = "127.0.0.1:7070",
    once: bool = false,
    policy: RuntimePolicy = .{},
    gpu: gpu.Capabilities = .{},
    accelerators: accelerator.Capabilities = accelerator.defaultCapabilities(),
};

pub const RuntimePolicy = struct {
    max_request_body_bytes: usize = 64 * 1024,
    max_invoke_args: usize = 16,
    max_stdin_bytes: usize = 64 * 1024,
    max_stdout_bytes: usize = 64 * 1024,
    max_stderr_bytes: usize = 64 * 1024,
    max_invoke_duration_ns: u64 = 5 * 1_000_000_000,
};

const ActivityKind = enum {
    workload_check,
    workload_deploy,
    workload_invoke,
};

const ActivityStatus = enum {
    accepted,
    rejected,
    failed,
};

const Activity = struct {
    id: u64,
    kind: ActivityKind,
    status: ActivityStatus,
    path: []const u8,
    detail: []const u8,

    fn deinit(self: *Activity, allocator: std.mem.Allocator) void {
        allocator.free(self.path);
        allocator.free(self.detail);
        self.* = undefined;
    }
};

const WorkloadStatus = enum {
    accepted,
    loaded,
    failed,
};

const RegisteredWorkload = struct {
    id: u64,
    status: WorkloadStatus,
    path: []const u8,
    name: []const u8,
    profile: []const u8,
    entrypoint: []const u8,
    wasm_path: []const u8,
    model_path: []const u8,
    memory_bytes: usize,
    last_error: []const u8,
    activity_id: u64,

    fn deinit(self: *RegisteredWorkload, allocator: std.mem.Allocator) void {
        allocator.free(self.path);
        allocator.free(self.name);
        allocator.free(self.profile);
        allocator.free(self.entrypoint);
        allocator.free(self.wasm_path);
        allocator.free(self.model_path);
        allocator.free(self.last_error);
        self.* = undefined;
    }

    fn setLastError(
        self: *RegisteredWorkload,
        allocator: std.mem.Allocator,
        error_name: []const u8,
    ) !void {
        const owned = try allocator.dupe(u8, error_name);
        allocator.free(self.last_error);
        self.last_error = owned;
    }
};

const WorkloadCandidate = struct {
    path: []const u8,
    name: []const u8,
    profile: []const u8,
    entrypoint: []const u8,
    wasm_path: []const u8,
    model_path: []const u8,
    memory_bytes: usize,

    fn deinit(self: *WorkloadCandidate, allocator: std.mem.Allocator) void {
        allocator.free(self.path);
        allocator.free(self.name);
        allocator.free(self.profile);
        allocator.free(self.entrypoint);
        allocator.free(self.wasm_path);
        allocator.free(self.model_path);
        self.* = undefined;
    }
};

pub const State = struct {
    next_activity_id: u64 = 1,
    next_workload_id: u64 = 1,
    activities: std.ArrayList(Activity) = .empty,
    workloads: std.ArrayList(RegisteredWorkload) = .empty,
    telemetry_store: telemetry.Store = .{},

    pub fn deinit(self: *State, allocator: std.mem.Allocator) void {
        self.telemetry_store.deinit(allocator);

        for (self.workloads.items) |*registered| {
            registered.deinit(allocator);
        }
        self.workloads.deinit(allocator);

        for (self.activities.items) |*activity| {
            activity.deinit(allocator);
        }
        self.activities.deinit(allocator);
        self.* = undefined;
    }

    fn record(
        self: *State,
        allocator: std.mem.Allocator,
        kind: ActivityKind,
        status: ActivityStatus,
        path: []const u8,
        detail: []const u8,
    ) !u64 {
        const owned_path = try allocator.dupe(u8, path);
        errdefer allocator.free(owned_path);

        const owned_detail = try allocator.dupe(u8, detail);
        errdefer allocator.free(owned_detail);

        const id = self.next_activity_id;
        self.next_activity_id += 1;

        try self.activities.append(allocator, .{
            .id = id,
            .kind = kind,
            .status = status,
            .path = owned_path,
            .detail = owned_detail,
        });

        return id;
    }

    fn findWorkloadByPath(self: *State, path: []const u8) ?*RegisteredWorkload {
        for (self.workloads.items) |*registered| {
            if (std.mem.eql(u8, registered.path, path)) return registered;
        }

        return null;
    }

    fn findWorkloadById(self: *State, id: u64) ?*RegisteredWorkload {
        for (self.workloads.items) |*registered| {
            if (registered.id == id) return registered;
        }

        return null;
    }

    fn registerWorkload(
        self: *State,
        allocator: std.mem.Allocator,
        candidate: WorkloadCandidate,
        activity_id: u64,
    ) !u64 {
        const owned_path = try allocator.dupe(u8, candidate.path);
        errdefer allocator.free(owned_path);

        const owned_name = try allocator.dupe(u8, candidate.name);
        errdefer allocator.free(owned_name);

        const owned_profile = try allocator.dupe(u8, candidate.profile);
        errdefer allocator.free(owned_profile);

        const owned_entrypoint = try allocator.dupe(u8, candidate.entrypoint);
        errdefer allocator.free(owned_entrypoint);

        const owned_wasm_path = try allocator.dupe(u8, candidate.wasm_path);
        errdefer allocator.free(owned_wasm_path);

        const owned_model_path = try allocator.dupe(u8, candidate.model_path);
        errdefer allocator.free(owned_model_path);

        const owned_last_error = try allocator.dupe(u8, "");
        errdefer allocator.free(owned_last_error);

        const id = self.next_workload_id;
        self.next_workload_id += 1;

        try self.workloads.append(allocator, .{
            .id = id,
            .status = .accepted,
            .path = owned_path,
            .name = owned_name,
            .profile = owned_profile,
            .entrypoint = owned_entrypoint,
            .wasm_path = owned_wasm_path,
            .model_path = owned_model_path,
            .memory_bytes = candidate.memory_bytes,
            .last_error = owned_last_error,
            .activity_id = activity_id,
        });

        return id;
    }
};

const Status = struct {
    code: u16,
    reason: []const u8,

    const ok: Status = .{ .code = 200, .reason = "OK" };
    const bad_request: Status = .{ .code = 400, .reason = "Bad Request" };
    const not_found: Status = .{ .code = 404, .reason = "Not Found" };
    const method_not_allowed: Status = .{ .code = 405, .reason = "Method Not Allowed" };
};

const RequestLine = struct {
    method: []const u8,
    target: []const u8,
};

pub fn serve(allocator: std.mem.Allocator, config: Config) !void {
    const io = std.Options.debug_io;
    var address = try parseListenAddress(config.listen);
    var server = try address.listen(io, .{ .reuse_address = true });
    defer server.deinit(io);

    var state = State{};
    defer state.deinit(allocator);

    std.debug.print("zug agent listening on {f}\n", .{server.socket.address});

    while (true) {
        const stream = try server.accept(io);
        try handleConnection(allocator, io, stream, config, &state);
        if (config.once) break;
    }
}

fn handleConnection(
    allocator: std.mem.Allocator,
    io: std.Io,
    stream: net.Stream,
    config: Config,
    state: *State,
) !void {
    defer stream.close(io);

    var read_buffer: [4096]u8 = undefined;
    var reader_state = stream.reader(io, &read_buffer);
    var request_buffer: [8192]u8 = undefined;
    const request = try readHttpRequest(&reader_state.interface, &request_buffer);

    const response = try responseForRequestWithState(allocator, request, config, state);
    defer allocator.free(response);

    var write_buffer: [4096]u8 = undefined;
    var writer_state = stream.writer(io, &write_buffer);
    try writer_state.interface.writeAll(response);
    try writer_state.interface.flush();
}

fn readHttpRequest(reader: *std.Io.Reader, buffer: []u8) ![]const u8 {
    var len: usize = 0;
    var required_len: ?usize = null;

    while (len < buffer.len) {
        const read_count = try reader.readSliceShort(buffer[len .. len + 1]);
        if (read_count == 0) break;
        len += read_count;

        const request = buffer[0..len];
        if (required_len == null) {
            if (headerBodyStart(request)) |body_start| {
                const content_length = try requestContentLength(request);
                const total_len = try std.math.add(usize, body_start, content_length);
                if (total_len > buffer.len) return error.HttpRequestTooLarge;
                required_len = total_len;
            }
        }

        if (required_len) |total_len| {
            if (len >= total_len) return buffer[0..total_len];
        }
    }

    if (len == buffer.len) return error.HttpRequestTooLarge;
    return buffer[0..len];
}

pub fn responseForRequest(
    allocator: std.mem.Allocator,
    request: []const u8,
    config: Config,
) ![]u8 {
    var state = State{};
    defer state.deinit(allocator);

    return responseForRequestWithState(allocator, request, config, &state);
}

pub fn responseForRequestWithState(
    allocator: std.mem.Allocator,
    request: []const u8,
    config: Config,
    state: *State,
) ![]u8 {
    const line = parseRequestLine(request) catch {
        return jsonResponse(allocator, .bad_request, "{\"error\":\"bad_request\"}");
    };

    const path = requestPath(line.target);
    if (std.mem.eql(u8, line.method, "GET")) {
        if (std.mem.eql(u8, path, "/") or std.mem.eql(u8, path, "/health")) {
            const body = try healthJson(allocator, config);
            defer allocator.free(body);
            return jsonResponse(allocator, .ok, body);
        }

        if (std.mem.eql(u8, path, "/capabilities")) {
            const body = try capabilitiesJson(allocator, config);
            defer allocator.free(body);
            return jsonResponse(allocator, .ok, body);
        }

        if (std.mem.eql(u8, path, "/activity")) {
            const body = try activityJson(allocator, state.*);
            defer allocator.free(body);
            return jsonResponse(allocator, .ok, body);
        }

        if (std.mem.eql(u8, path, "/telemetry")) {
            const body = try telemetryJson(allocator, state.telemetry_store);
            defer allocator.free(body);
            return jsonResponse(allocator, .ok, body);
        }

        if (std.mem.eql(u8, path, "/telemetry/events")) {
            const body = try telemetryEventsJson(allocator, state.telemetry_store);
            defer allocator.free(body);
            return jsonResponse(allocator, .ok, body);
        }

        if (std.mem.eql(u8, path, "/telemetry/metrics")) {
            const body = try telemetryMetricsJson(allocator, state.telemetry_store);
            defer allocator.free(body);
            return jsonResponse(allocator, .ok, body);
        }

        if (std.mem.eql(u8, path, "/telemetry/traces")) {
            const body = try telemetryTracesJson(allocator, state.telemetry_store);
            defer allocator.free(body);
            return jsonResponse(allocator, .ok, body);
        }

        if (std.mem.eql(u8, path, "/telemetry/otlp/logs")) {
            const body = try otlpLogsJson(allocator, config, state.telemetry_store);
            defer allocator.free(body);
            return jsonResponse(allocator, .ok, body);
        }

        if (std.mem.eql(u8, path, "/telemetry/otlp/metrics")) {
            const body = try otlpMetricsJson(allocator, config, state.telemetry_store);
            defer allocator.free(body);
            return jsonResponse(allocator, .ok, body);
        }

        if (std.mem.eql(u8, path, "/telemetry/otlp/traces")) {
            const body = try otlpTracesJson(allocator, config, state.telemetry_store);
            defer allocator.free(body);
            return jsonResponse(allocator, .ok, body);
        }

        if (std.mem.eql(u8, path, "/workloads")) {
            const body = try workloadsJson(allocator, state.*);
            defer allocator.free(body);
            return jsonResponse(allocator, .ok, body);
        }

        return jsonResponse(allocator, .not_found, "{\"error\":\"not_found\"}");
    }

    if (std.mem.eql(u8, line.method, "POST")) {
        if (std.mem.eql(u8, path, "/workloads/check")) {
            const workload_path = parseWorkloadCheckPath(requestBody(request)) catch {
                return jsonResponse(allocator, .bad_request, "{\"error\":\"invalid_workload_check_request\"}");
            };

            const body = try workloadCheckJson(allocator, workload_path, config, state);
            defer allocator.free(body);
            return jsonResponse(allocator, .ok, body);
        }

        if (std.mem.eql(u8, path, "/workloads/deploy")) {
            const workload_path = parseWorkloadCheckPath(requestBody(request)) catch {
                return jsonResponse(allocator, .bad_request, "{\"error\":\"invalid_workload_deploy_request\"}");
            };

            const body = try workloadDeployJson(allocator, workload_path, config, state);
            defer allocator.free(body);
            return jsonResponse(allocator, .ok, body);
        }

        if (std.mem.eql(u8, path, "/telemetry/events")) {
            const input = parseTelemetryEventRequest(requestBody(request), config.policy) catch {
                return jsonResponse(allocator, .bad_request, "{\"error\":\"invalid_telemetry_event\"}");
            };

            const event = state.telemetry_store.recordEvent(allocator, input, monotonicNowNs()) catch {
                return jsonResponse(allocator, .bad_request, "{\"error\":\"invalid_telemetry_event\"}");
            };

            const body = try telemetryEventAcceptedJson(allocator, event);
            defer allocator.free(body);
            return jsonResponse(allocator, .ok, body);
        }

        if (std.mem.eql(u8, path, "/telemetry/metrics")) {
            const input = parseTelemetryMetricRequest(requestBody(request), config.policy) catch {
                return jsonResponse(allocator, .bad_request, "{\"error\":\"invalid_telemetry_metric\"}");
            };

            const metric = state.telemetry_store.recordMetric(allocator, input, monotonicNowNs()) catch {
                return jsonResponse(allocator, .bad_request, "{\"error\":\"invalid_telemetry_metric\"}");
            };

            const body = try telemetryMetricAcceptedJson(allocator, metric);
            defer allocator.free(body);
            return jsonResponse(allocator, .ok, body);
        }

        if (std.mem.eql(u8, path, "/telemetry/traces")) {
            const input = parseTelemetryTraceRequest(requestBody(request), config.policy) catch {
                return jsonResponse(allocator, .bad_request, "{\"error\":\"invalid_telemetry_trace\"}");
            };

            const span = state.telemetry_store.recordTrace(allocator, input) catch {
                return jsonResponse(allocator, .bad_request, "{\"error\":\"invalid_telemetry_trace\"}");
            };

            const body = try telemetryTraceAcceptedJson(allocator, span);
            defer allocator.free(body);
            return jsonResponse(allocator, .ok, body);
        }

        if (std.mem.eql(u8, path, "/v1/logs")) {
            ingestOtlpLogs(allocator, requestBody(request), config.policy, &state.telemetry_store) catch {
                return jsonResponse(allocator, .bad_request, "{\"error\":\"invalid_otlp_logs\"}");
            };

            return jsonResponse(allocator, .ok, "{}");
        }

        if (std.mem.eql(u8, path, "/v1/metrics")) {
            ingestOtlpMetrics(allocator, requestBody(request), config.policy, &state.telemetry_store) catch {
                return jsonResponse(allocator, .bad_request, "{\"error\":\"invalid_otlp_metrics\"}");
            };

            return jsonResponse(allocator, .ok, "{}");
        }

        if (std.mem.eql(u8, path, "/v1/traces")) {
            ingestOtlpTraces(allocator, requestBody(request), config.policy, &state.telemetry_store) catch {
                return jsonResponse(allocator, .bad_request, "{\"error\":\"invalid_otlp_traces\"}");
            };

            return jsonResponse(allocator, .ok, "{}");
        }

        if (parseInvokeWorkloadId(path)) |workload_id| {
            var invoke_request = parseInvokeRequest(allocator, requestBody(request), config.policy) catch {
                return jsonResponse(allocator, .bad_request, "{\"error\":\"invalid_workload_invoke_request\"}");
            };
            defer invoke_request.deinit(allocator);

            const body = try workloadInvokeJson(allocator, workload_id, invoke_request, config.policy, config.gpu, config.accelerators, state);
            defer allocator.free(body);
            return jsonResponse(allocator, .ok, body);
        } else |err| switch (err) {
            error.NotInvokePath => {},
            else => return jsonResponse(allocator, .bad_request, "{\"error\":\"invalid_workload_invoke_path\"}"),
        }

        return jsonResponse(allocator, .not_found, "{\"error\":\"not_found\"}");
    }

    return jsonResponse(allocator, .method_not_allowed, "{\"error\":\"method_not_allowed\"}");
}

const OnnxAnalysis = struct {
    path: []const u8,
    byte_len: usize = 0,
    read_error: ?[]const u8 = null,
    decode_error: ?[]const u8 = null,
    model: ?onnx.ModelProto = null,
    report: ?capabilities.Report = null,
    ok: bool = false,

    fn deinit(self: *OnnxAnalysis, allocator: std.mem.Allocator) void {
        if (self.report) |*report| {
            report.deinit(allocator);
        }
        if (self.model) |*model| {
            model.deinit(allocator);
        }
        self.* = undefined;
    }
};

const InvokeResult = struct {
    result: ?wasm_interpreter.Value = null,
    stdout: []const u8,
    stderr: []const u8,
    exit_code: ?u32 = null,
    memory_bytes: usize,
    duration_ns: u64,

    fn deinit(self: *InvokeResult, allocator: std.mem.Allocator) void {
        allocator.free(self.stdout);
        allocator.free(self.stderr);
        self.* = undefined;
    }
};

const InvokeRequest = struct {
    args: []const u32 = &.{},
    stdin: []const u8 = "",

    fn deinit(self: *InvokeRequest, allocator: std.mem.Allocator) void {
        allocator.free(self.args);
        allocator.free(self.stdin);
        self.* = undefined;
    }
};

fn headerBodyStart(request: []const u8) ?usize {
    if (std.mem.indexOf(u8, request, "\r\n\r\n")) |index| return index + 4;
    if (std.mem.indexOf(u8, request, "\n\n")) |index| return index + 2;
    return null;
}

fn requestHeaderBytes(request: []const u8) []const u8 {
    if (std.mem.indexOf(u8, request, "\r\n\r\n")) |index| return request[0..index];
    if (std.mem.indexOf(u8, request, "\n\n")) |index| return request[0..index];
    return request;
}

fn requestBody(request: []const u8) []const u8 {
    const body_start = headerBodyStart(request) orelse return "";
    return request[body_start..];
}

fn requestContentLength(request: []const u8) !usize {
    const headers = requestHeaderBytes(request);
    var lines = std.mem.splitScalar(u8, headers, '\n');
    _ = lines.next();

    while (lines.next()) |raw_line| {
        const line = std.mem.trim(u8, raw_line, " \t\r");
        if (line.len == 0) continue;

        const colon = std.mem.indexOfScalar(u8, line, ':') orelse continue;
        const name = std.mem.trim(u8, line[0..colon], " \t\r");
        if (!std.ascii.eqlIgnoreCase(name, "content-length")) continue;

        const value = std.mem.trim(u8, line[colon + 1 ..], " \t\r");
        return std.fmt.parseInt(usize, value, 10) catch error.InvalidContentLength;
    }

    return 0;
}

fn parseWorkloadCheckPath(body: []const u8) ![]const u8 {
    const path = try jsonStringField(body, "path");
    if (path.len == 0) return error.MissingWorkloadPath;
    return path;
}

fn parseInvokeWorkloadId(path: []const u8) !u64 {
    const prefix = "/workloads/";
    const suffix = "/invoke";

    if (!std.mem.startsWith(u8, path, prefix)) return error.NotInvokePath;
    if (!std.mem.endsWith(u8, path, suffix)) return error.NotInvokePath;
    if (path.len <= prefix.len + suffix.len) return error.InvalidWorkloadId;

    const id_text = path[prefix.len .. path.len - suffix.len];
    if (std.mem.indexOfScalar(u8, id_text, '/') != null) return error.InvalidWorkloadId;
    return std.fmt.parseInt(u64, id_text, 10) catch error.InvalidWorkloadId;
}

fn parseInvokeRequest(
    allocator: std.mem.Allocator,
    body: []const u8,
    policy: RuntimePolicy,
) !InvokeRequest {
    if (body.len > policy.max_request_body_bytes) return error.InvokeRequestBodyTooLarge;

    if (std.mem.trim(u8, body, " \t\r\n").len == 0) {
        return .{
            .args = try allocator.alloc(u32, 0),
            .stdin = try allocator.dupe(u8, ""),
        };
    }

    const args = jsonU32ArrayField(allocator, body, "args") catch |err| switch (err) {
        error.MissingJsonField => try allocator.alloc(u32, 0),
        else => return err,
    };
    errdefer allocator.free(args);
    if (args.len > policy.max_invoke_args) return error.InvokeArgLimitExceeded;

    const stdin_value = jsonStringField(body, "stdin") catch |err| switch (err) {
        error.MissingJsonField => "",
        else => return err,
    };
    if (stdin_value.len > policy.max_stdin_bytes) return error.InvokeStdinTooLarge;

    const stdin = try allocator.dupe(u8, stdin_value);
    errdefer allocator.free(stdin);

    return .{
        .args = args,
        .stdin = stdin,
    };
}

fn parseTelemetryEventRequest(body: []const u8, policy: RuntimePolicy) !telemetry.EventInput {
    if (body.len > policy.max_request_body_bytes) return error.TelemetryRequestBodyTooLarge;

    const source = jsonStringField(body, "source") catch |err| switch (err) {
        error.MissingJsonField => "agent",
        else => return err,
    };
    const severity_name = jsonStringField(body, "severity") catch |err| switch (err) {
        error.MissingJsonField => "info",
        else => return err,
    };

    return .{
        .source = source,
        .kind = try jsonStringField(body, "kind"),
        .message = try jsonStringField(body, "message"),
        .severity = telemetry.parseSeverity(severity_name) orelse return error.InvalidTelemetrySeverity,
    };
}

fn parseTelemetryMetricRequest(body: []const u8, policy: RuntimePolicy) !telemetry.MetricInput {
    if (body.len > policy.max_request_body_bytes) return error.TelemetryRequestBodyTooLarge;

    const source = jsonStringField(body, "source") catch |err| switch (err) {
        error.MissingJsonField => "agent",
        else => return err,
    };
    const unit = jsonStringField(body, "unit") catch |err| switch (err) {
        error.MissingJsonField => "",
        else => return err,
    };

    return .{
        .source = source,
        .name = try jsonStringField(body, "name"),
        .value = try jsonF64Field(body, "value"),
        .unit = unit,
    };
}

fn parseTelemetryTraceRequest(body: []const u8, policy: RuntimePolicy) !telemetry.TraceInput {
    if (body.len > policy.max_request_body_bytes) return error.TelemetryRequestBodyTooLarge;

    const source = jsonStringField(body, "source") catch |err| switch (err) {
        error.MissingJsonField => "agent",
        else => return err,
    };
    const parent_span_id = jsonStringField(body, "parent_span_id") catch |err| switch (err) {
        error.MissingJsonField => "",
        else => return err,
    };
    const kind_name = jsonStringField(body, "kind") catch |err| switch (err) {
        error.MissingJsonField => "internal",
        else => return err,
    };
    const status_name = jsonStringField(body, "status") catch |err| switch (err) {
        error.MissingJsonField => "unset",
        else => return err,
    };

    return .{
        .source = source,
        .trace_id = try jsonStringField(body, "trace_id"),
        .span_id = try jsonStringField(body, "span_id"),
        .parent_span_id = parent_span_id,
        .name = try jsonStringField(body, "name"),
        .kind = telemetry.parseSpanKind(kind_name) orelse return error.InvalidTelemetrySpanKind,
        .start_time_ns = try jsonU64Field(body, "start_time_ns"),
        .end_time_ns = try jsonU64Field(body, "end_time_ns"),
        .status = telemetry.parseSpanStatus(status_name) orelse return error.InvalidTelemetrySpanStatus,
    };
}

fn ingestOtlpLogs(
    allocator: std.mem.Allocator,
    body: []const u8,
    policy: RuntimePolicy,
    store: *telemetry.Store,
) !void {
    if (body.len > policy.max_request_body_bytes) return error.TelemetryRequestBodyTooLarge;

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, body, .{
        .ignore_unknown_fields = true,
    });
    defer parsed.deinit();

    const root = parsed.value;
    const resource_logs = jsonArrayField(root, "resourceLogs") orelse return error.MissingOtlpResourceLogs;
    var accepted: usize = 0;

    for (resource_logs.items) |resource_log| {
        const source = otlpSource(resource_log);
        const scope_logs = jsonArrayField(resource_log, "scopeLogs") orelse continue;

        for (scope_logs.items) |scope_log| {
            const log_records = jsonArrayField(scope_log, "logRecords") orelse continue;

            for (log_records.items) |record| {
                const message = otlpLogBody(record) orelse "otel log record";
                const kind = otlpAttributeString(record, "event.name") orelse "otel.log";
                const severity = otlpLogSeverity(record);
                _ = try store.recordEvent(allocator, .{
                    .source = source,
                    .kind = kind,
                    .message = message,
                    .severity = severity,
                }, otlpTimestamp(record) orelse monotonicNowNs());
                accepted += 1;
            }
        }
    }

    if (accepted == 0) return error.EmptyOtlpLogs;
}

fn ingestOtlpMetrics(
    allocator: std.mem.Allocator,
    body: []const u8,
    policy: RuntimePolicy,
    store: *telemetry.Store,
) !void {
    if (body.len > policy.max_request_body_bytes) return error.TelemetryRequestBodyTooLarge;

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, body, .{
        .ignore_unknown_fields = true,
    });
    defer parsed.deinit();

    const root = parsed.value;
    const resource_metrics = jsonArrayField(root, "resourceMetrics") orelse return error.MissingOtlpResourceMetrics;
    var accepted: usize = 0;

    for (resource_metrics.items) |resource_metric| {
        const source = otlpSource(resource_metric);
        const scope_metrics = jsonArrayField(resource_metric, "scopeMetrics") orelse continue;

        for (scope_metrics.items) |scope_metric| {
            const metrics = jsonArrayField(scope_metric, "metrics") orelse continue;

            for (metrics.items) |metric| {
                const name = jsonStringFieldValue(metric, "name") orelse continue;
                const unit = jsonStringFieldValue(metric, "unit") orelse "";
                const datapoints = otlpMetricDataPoints(metric) orelse continue;

                for (datapoints.items) |datapoint| {
                    const value = otlpMetricValue(datapoint) orelse continue;
                    _ = try store.recordMetric(allocator, .{
                        .source = source,
                        .name = name,
                        .value = value,
                        .unit = unit,
                    }, otlpTimestamp(datapoint) orelse monotonicNowNs());
                    accepted += 1;
                }
            }
        }
    }

    if (accepted == 0) return error.EmptyOtlpMetrics;
}

fn ingestOtlpTraces(
    allocator: std.mem.Allocator,
    body: []const u8,
    policy: RuntimePolicy,
    store: *telemetry.Store,
) !void {
    if (body.len > policy.max_request_body_bytes) return error.TelemetryRequestBodyTooLarge;

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, body, .{
        .ignore_unknown_fields = true,
    });
    defer parsed.deinit();

    const root = parsed.value;
    const resource_spans = jsonArrayField(root, "resourceSpans") orelse return error.MissingOtlpResourceSpans;
    var accepted: usize = 0;

    for (resource_spans.items) |resource_span| {
        const source = otlpSource(resource_span);
        const scope_spans = jsonArrayField(resource_span, "scopeSpans") orelse continue;

        for (scope_spans.items) |scope_span| {
            const spans = jsonArrayField(scope_span, "spans") orelse continue;

            for (spans.items) |span| {
                const trace_id = jsonStringFieldValue(span, "traceId") orelse continue;
                const span_id = jsonStringFieldValue(span, "spanId") orelse continue;
                const name = jsonStringFieldValue(span, "name") orelse continue;

                _ = try store.recordTrace(allocator, .{
                    .source = source,
                    .trace_id = trace_id,
                    .span_id = span_id,
                    .parent_span_id = jsonStringFieldValue(span, "parentSpanId") orelse "",
                    .name = name,
                    .kind = otlpSpanKind(span),
                    .start_time_ns = otlpStartTimestamp(span) orelse 0,
                    .end_time_ns = otlpEndTimestamp(span) orelse 0,
                    .status = otlpSpanStatus(span),
                });
                accepted += 1;
            }
        }
    }

    if (accepted == 0) return error.EmptyOtlpTraces;
}

fn jsonStringField(body: []const u8, field_name: []const u8) ![]const u8 {
    var index: usize = 0;
    while (index < body.len) {
        if (body[index] != '"') {
            index += 1;
            continue;
        }

        const key_start = index + 1;
        const key_end = try jsonStringEnd(body, key_start);
        const key = body[key_start..key_end];
        index = skipJsonWhitespace(body, key_end + 1);

        if (index >= body.len or body[index] != ':') continue;
        index = skipJsonWhitespace(body, index + 1);

        if (!std.mem.eql(u8, key, field_name)) continue;
        if (index >= body.len or body[index] != '"') return error.ExpectedJsonString;

        const value_start = index + 1;
        const value_end = try jsonStringEnd(body, value_start);
        return try decodeJsonString(body[value_start..value_end]);
    }

    return error.MissingJsonField;
}

fn jsonF64Field(body: []const u8, field_name: []const u8) !f64 {
    const raw = try jsonNumberField(body, field_name);
    return std.fmt.parseFloat(f64, raw) catch error.InvalidJsonNumber;
}

fn jsonU64Field(body: []const u8, field_name: []const u8) !u64 {
    const raw = try jsonNumberField(body, field_name);
    return std.fmt.parseInt(u64, raw, 10) catch error.InvalidJsonNumber;
}

fn jsonNumberField(body: []const u8, field_name: []const u8) ![]const u8 {
    var index: usize = 0;
    while (index < body.len) {
        if (body[index] != '"') {
            index += 1;
            continue;
        }

        const key_start = index + 1;
        const key_end = try jsonStringEnd(body, key_start);
        const key = body[key_start..key_end];
        index = skipJsonWhitespace(body, key_end + 1);

        if (index >= body.len or body[index] != ':') continue;
        index = skipJsonWhitespace(body, index + 1);

        if (!std.mem.eql(u8, key, field_name)) continue;

        const value_start = index;
        while (index < body.len) : (index += 1) {
            switch (body[index]) {
                '0'...'9', '-', '+', '.', 'e', 'E' => {},
                else => break,
            }
        }

        if (index == value_start) return error.ExpectedJsonNumber;
        return body[value_start..index];
    }

    return error.MissingJsonField;
}

fn decodeJsonString(value: []const u8) ![]const u8 {
    var index: usize = 0;
    while (index < value.len) : (index += 1) {
        if (value[index] == '\\') return error.UnsupportedJsonEscape;
    }

    return value;
}

fn jsonU32ArrayField(allocator: std.mem.Allocator, body: []const u8, field_name: []const u8) ![]u32 {
    var index: usize = 0;
    while (index < body.len) {
        if (body[index] != '"') {
            index += 1;
            continue;
        }

        const key_start = index + 1;
        const key_end = try jsonStringEnd(body, key_start);
        const key = body[key_start..key_end];
        index = skipJsonWhitespace(body, key_end + 1);

        if (index >= body.len or body[index] != ':') continue;
        index = skipJsonWhitespace(body, index + 1);

        if (!std.mem.eql(u8, key, field_name)) continue;
        if (index >= body.len or body[index] != '[') return error.ExpectedJsonArray;

        return parseJsonU32Array(allocator, body, index + 1);
    }

    return error.MissingJsonField;
}

fn parseJsonU32Array(allocator: std.mem.Allocator, body: []const u8, start: usize) ![]u32 {
    var values: std.ArrayList(u32) = .empty;
    errdefer values.deinit(allocator);

    var index = skipJsonWhitespace(body, start);
    if (index < body.len and body[index] == ']') {
        return try values.toOwnedSlice(allocator);
    }

    while (index < body.len) {
        const value_start = index;
        if (body[index] == '-') return error.NegativeInvokeArg;
        while (index < body.len and std.ascii.isDigit(body[index])) : (index += 1) {}
        if (index == value_start) return error.ExpectedJsonNumber;

        const value = try std.fmt.parseInt(u32, body[value_start..index], 10);
        try values.append(allocator, value);

        index = skipJsonWhitespace(body, index);
        if (index >= body.len) return error.UnterminatedJsonArray;
        if (body[index] == ']') return try values.toOwnedSlice(allocator);
        if (body[index] != ',') return error.ExpectedJsonArrayComma;
        index = skipJsonWhitespace(body, index + 1);
    }

    return error.UnterminatedJsonArray;
}

fn jsonObjectField(value: std.json.Value, field_name: []const u8) ?std.json.Value {
    if (value != .object) return null;
    return value.object.get(field_name);
}

fn jsonArrayField(value: std.json.Value, field_name: []const u8) ?std.json.Array {
    const field = jsonObjectField(value, field_name) orelse return null;
    if (field != .array) return null;
    return field.array;
}

fn jsonStringFieldValue(value: std.json.Value, field_name: []const u8) ?[]const u8 {
    const field = jsonObjectField(value, field_name) orelse return null;
    return jsonValueAsString(field);
}

fn jsonValueAsString(value: std.json.Value) ?[]const u8 {
    return switch (value) {
        .string => |actual| actual,
        .number_string => |actual| actual,
        else => null,
    };
}

fn jsonValueAsF64(value: std.json.Value) ?f64 {
    return switch (value) {
        .float => |actual| actual,
        .integer => |actual| @floatFromInt(actual),
        .number_string => |actual| std.fmt.parseFloat(f64, actual) catch null,
        .string => |actual| std.fmt.parseFloat(f64, actual) catch null,
        else => null,
    };
}

fn jsonValueAsU64(value: std.json.Value) ?u64 {
    return switch (value) {
        .integer => |actual| if (actual >= 0) @intCast(actual) else null,
        .number_string => |actual| std.fmt.parseInt(u64, actual, 10) catch null,
        .string => |actual| std.fmt.parseInt(u64, actual, 10) catch null,
        else => null,
    };
}

fn otlpSource(resource_container: std.json.Value) []const u8 {
    const resource = jsonObjectField(resource_container, "resource") orelse return "otel";
    return otlpAttributeString(resource, "service.name") orelse "otel";
}

fn otlpAttributeString(container: std.json.Value, key_name: []const u8) ?[]const u8 {
    const attributes = jsonArrayField(container, "attributes") orelse return null;
    for (attributes.items) |attribute| {
        const key = jsonStringFieldValue(attribute, "key") orelse continue;
        if (!std.mem.eql(u8, key, key_name)) continue;
        const value = jsonObjectField(attribute, "value") orelse return null;
        return otlpAnyValueString(value);
    }
    return null;
}

fn otlpAnyValueString(value: std.json.Value) ?[]const u8 {
    if (value == .string or value == .number_string) return jsonValueAsString(value);
    if (value != .object) return null;
    if (jsonStringFieldValue(value, "stringValue")) |actual| return actual;
    if (jsonStringFieldValue(value, "intValue")) |actual| return actual;
    if (jsonStringFieldValue(value, "doubleValue")) |actual| return actual;
    if (jsonStringFieldValue(value, "boolValue")) |actual| return actual;
    return null;
}

fn otlpLogBody(record: std.json.Value) ?[]const u8 {
    const body = jsonObjectField(record, "body") orelse return null;
    return otlpAnyValueString(body);
}

fn otlpLogSeverity(record: std.json.Value) telemetry.Severity {
    if (jsonStringFieldValue(record, "severityText")) |severity_text| {
        if (std.ascii.eqlIgnoreCase(severity_text, "error") or
            std.ascii.eqlIgnoreCase(severity_text, "err") or
            std.ascii.eqlIgnoreCase(severity_text, "fatal"))
        {
            return .err;
        }
        if (std.ascii.eqlIgnoreCase(severity_text, "warn") or
            std.ascii.eqlIgnoreCase(severity_text, "warning"))
        {
            return .warn;
        }
    }

    const severity_number_value = jsonObjectField(record, "severityNumber") orelse return .info;
    const severity_number = jsonValueAsU64(severity_number_value) orelse return .info;
    if (severity_number >= 17) return .err;
    if (severity_number >= 13) return .warn;
    return .info;
}

fn otlpTimestamp(value: std.json.Value) ?u64 {
    const time_value = jsonObjectField(value, "timeUnixNano") orelse
        jsonObjectField(value, "observedTimeUnixNano") orelse
        return null;
    return jsonValueAsU64(time_value);
}

fn otlpStartTimestamp(value: std.json.Value) ?u64 {
    const time_value = jsonObjectField(value, "startTimeUnixNano") orelse return null;
    return jsonValueAsU64(time_value);
}

fn otlpEndTimestamp(value: std.json.Value) ?u64 {
    const time_value = jsonObjectField(value, "endTimeUnixNano") orelse return null;
    return jsonValueAsU64(time_value);
}

fn otlpSpanKind(span: std.json.Value) telemetry.SpanKind {
    const value = jsonObjectField(span, "kind") orelse return .unspecified;
    const kind = jsonValueAsU64(value) orelse return .unspecified;
    return switch (kind) {
        1 => .internal,
        2 => .server,
        3 => .client,
        4 => .producer,
        5 => .consumer,
        else => .unspecified,
    };
}

fn otlpSpanStatus(span: std.json.Value) telemetry.SpanStatus {
    const status = jsonObjectField(span, "status") orelse return .unset;
    const code_value = jsonObjectField(status, "code") orelse return .unset;
    const code = jsonValueAsU64(code_value) orelse return .unset;
    return switch (code) {
        1 => .ok,
        2 => .err,
        else => .unset,
    };
}

fn otlpMetricDataPoints(metric: std.json.Value) ?std.json.Array {
    if (jsonObjectField(metric, "gauge")) |gauge| {
        if (jsonArrayField(gauge, "dataPoints")) |points| return points;
    }
    if (jsonObjectField(metric, "sum")) |sum| {
        if (jsonArrayField(sum, "dataPoints")) |points| return points;
    }
    return null;
}

fn otlpMetricValue(datapoint: std.json.Value) ?f64 {
    if (jsonObjectField(datapoint, "asDouble")) |value| return jsonValueAsF64(value);
    if (jsonObjectField(datapoint, "asInt")) |value| return jsonValueAsF64(value);
    return null;
}

fn jsonStringEnd(body: []const u8, start: usize) !usize {
    var index = start;
    var escaped = false;
    while (index < body.len) : (index += 1) {
        if (escaped) {
            escaped = false;
            continue;
        }

        switch (body[index]) {
            '"' => return index,
            '\\' => escaped = true,
            else => {},
        }
    }

    return error.UnterminatedJsonString;
}

fn skipJsonWhitespace(body: []const u8, start: usize) usize {
    var index = start;
    while (index < body.len) : (index += 1) {
        switch (body[index]) {
            ' ', '\t', '\r', '\n' => {},
            else => return index,
        }
    }

    return index;
}

fn workloadCheckJson(
    allocator: std.mem.Allocator,
    path: []const u8,
    config: Config,
    state: *State,
) ![]u8 {
    var loaded = workload.Loaded.load(allocator, path) catch |err| {
        const activity_id = try state.record(allocator, .workload_check, .rejected, path, @errorName(err));
        return workloadLoadFailureJson(allocator, path, err, activity_id);
    };
    defer loaded.deinit(allocator);

    const requested_profile_name = loaded.manifest.targetName() orelse config.profile_name;
    const profile = target_profile.resolve(requested_profile_name) catch |err| {
        const activity_id = try state.record(allocator, .workload_check, .rejected, path, @errorName(err));
        return workloadProfileFailureJson(allocator, loaded, requested_profile_name, err, activity_id);
    };

    const manifest_error: ?[]const u8 = blk: {
        loaded.manifest.validateForTarget(profile) catch |err| break :blk @errorName(err);
        break :blk null;
    };

    var network_report = try network.analyze(allocator, loaded.manifest.network, profile.network);
    defer network_report.deinit(allocator);

    const wasm_bytes = std.Io.Dir.cwd().readFileAlloc(
        std.Options.debug_io,
        loaded.wasm_path,
        allocator,
        .limited(scope.max_wasm_bytes),
    ) catch |err| {
        const activity_id = try state.record(allocator, .workload_check, .rejected, path, @errorName(err));
        return workloadArtifactFailureJson(allocator, loaded, "wasm", err, activity_id);
    };
    defer allocator.free(wasm_bytes);

    var wasm_report = try wasm_compatibility.analyzeBytes(allocator, wasm_bytes, compatibilityTarget(profile), .{
        .initial_memory_bytes = loaded.manifest.requires.memory_bytes,
        .export_name = loaded.manifest.entrypointName(),
    });
    defer wasm_report.deinit(allocator);

    const missing_ml_import = loaded.manifest.requires.wasi_nn and !hasMlImport(wasm_report);

    var model_analysis: ?OnnxAnalysis = null;
    defer if (model_analysis) |*analysis| analysis.deinit(allocator);

    var model_size_error = false;
    if (loaded.model_path) |model_path| {
        model_analysis = analyzeOnnxPathDetailed(allocator, model_path) catch |err| .{
            .path = model_path,
            .read_error = @errorName(err),
        };
        model_size_error = model_analysis.?.read_error == null and model_analysis.?.byte_len > profile.max_model_bytes;
    }

    const model_ok = if (model_analysis) |analysis|
        analysis.ok and !model_size_error
    else
        !loaded.manifest.requires.wasi_nn;

    const ok = manifest_error == null and
        !missing_ml_import and
        network_report.supported() and
        wasm_report.supported() and
        model_ok;

    const activity_id = try state.record(
        allocator,
        .workload_check,
        if (ok) .accepted else .rejected,
        path,
        if (ok) "supported" else "unsupported",
    );

    return workloadCheckReportJson(
        allocator,
        loaded,
        profile,
        manifest_error,
        network_report,
        wasm_bytes.len,
        wasm_report,
        missing_ml_import,
        model_analysis,
        model_size_error,
        ok,
        activity_id,
    );
}

fn workloadDeployJson(
    allocator: std.mem.Allocator,
    path: []const u8,
    config: Config,
    state: *State,
) ![]u8 {
    var candidate = checkWorkloadDeployable(allocator, path, config) catch |err| {
        const activity_id = try state.record(allocator, .workload_deploy, .rejected, path, @errorName(err));
        return workloadDeployFailureJson(allocator, path, err, activity_id);
    };
    defer candidate.deinit(allocator);

    if (state.findWorkloadByPath(candidate.path)) |registered| {
        const activity_id = try state.record(allocator, .workload_deploy, .accepted, candidate.path, "already_registered");
        registered.activity_id = activity_id;
        return workloadDeploySuccessJson(allocator, registered.*, activity_id, true);
    }

    const activity_id = try state.record(allocator, .workload_deploy, .accepted, candidate.path, "accepted");
    const workload_id = try state.registerWorkload(allocator, candidate, activity_id);
    const registered = state.workloads.items[state.workloads.items.len - 1];
    std.debug.assert(registered.id == workload_id);

    return workloadDeploySuccessJson(allocator, registered, activity_id, false);
}

fn workloadInvokeJson(
    allocator: std.mem.Allocator,
    workload_id: u64,
    request: InvokeRequest,
    policy: RuntimePolicy,
    gpu_caps: gpu.Capabilities,
    accelerator_caps: accelerator.Capabilities,
    state: *State,
) ![]u8 {
    const registered = state.findWorkloadById(workload_id) orelse {
        return workloadInvokeMissingJson(allocator, workload_id);
    };

    var result = executeRegisteredWorkload(allocator, registered.*, request, policy, gpu_caps, accelerator_caps) catch |err| {
        registered.status = .failed;
        try registered.setLastError(allocator, @errorName(err));
        const activity_id = try state.record(allocator, .workload_invoke, .failed, registered.path, @errorName(err));
        registered.activity_id = activity_id;
        return workloadInvokeFailureJson(allocator, registered.*, err, activity_id);
    };
    defer result.deinit(allocator);

    registered.status = .loaded;
    try registered.setLastError(allocator, "");
    const activity_id = try state.record(allocator, .workload_invoke, .accepted, registered.path, "completed");
    registered.activity_id = activity_id;

    return workloadInvokeSuccessJson(allocator, registered.*, result, activity_id);
}

fn executeRegisteredWorkload(
    allocator: std.mem.Allocator,
    registered: RegisteredWorkload,
    request: InvokeRequest,
    policy: RuntimePolicy,
    gpu_caps: gpu.Capabilities,
    accelerator_caps: accelerator.Capabilities,
) !InvokeResult {
    const started_ns = monotonicNowNs();

    const wasm_bytes = try std.Io.Dir.cwd().readFileAlloc(
        std.Options.debug_io,
        registered.wasm_path,
        allocator,
        .limited(scope.max_wasm_bytes),
    );
    defer allocator.free(wasm_bytes);

    const model_bytes = if (registered.model_path.len != 0)
        try std.Io.Dir.cwd().readFileAlloc(
            std.Options.debug_io,
            registered.model_path,
            allocator,
            .limited(scope.max_model_bytes),
        )
    else
        null;
    defer if (model_bytes) |loaded| allocator.free(loaded);

    const profile = try target_profile.resolve(registered.profile);
    const runtime = wasm_runtime.Runtime.init(allocator);

    var parsed = try runtime.parseModule(wasm_bytes);
    defer parsed.deinit(allocator);

    var compatibility_report = try wasm_compatibility.analyzeModule(allocator, wasm_bytes, &parsed, compatibilityTarget(profile), .{
        .initial_memory_bytes = registered.memory_bytes,
        .export_name = registered.entrypoint,
    });
    defer compatibility_report.deinit(allocator);

    if (!compatibility_report.supported()) return error.WasmCompatibilityCheckFailed;

    var wasm_instance = try runtime.instantiate(&parsed, registered.memory_bytes);
    defer wasm_instance.deinit();

    var host = wasi_nn_abi.Host.initWithAccelerators(allocator, accelerator_caps);
    defer host.deinit();

    var surface = wasi_nn_abi.Surface.init(allocator, &host, &wasm_instance.memory);
    surface.setPreloadedModel(model_bytes);

    var resolver = wasm_imports.Resolver.init(&surface);
    resolver.stdin = request.stdin;
    resolver.gpu = gpu_caps;
    defer resolver.deinit();

    try wasm_instance.bindImports(&resolver);

    var interpreter = wasm_interpreter.Interpreter.init(&wasm_instance);
    try interpreter.runStart();

    var args: std.ArrayList(wasm_interpreter.Value) = .empty;
    defer args.deinit(allocator);

    for (request.args) |arg| {
        try args.append(allocator, .{ .i32 = arg });
    }

    const result = try interpreter.callExport(registered.entrypoint, args.items);

    if (resolver.stdout.items.len > policy.max_stdout_bytes) return error.InvokeStdoutTooLarge;
    if (resolver.stderr.items.len > policy.max_stderr_bytes) return error.InvokeStderrTooLarge;

    const stdout = try allocator.dupe(u8, resolver.stdout.items);
    errdefer allocator.free(stdout);

    const stderr = try allocator.dupe(u8, resolver.stderr.items);
    errdefer allocator.free(stderr);

    const finished_ns = monotonicNowNs();
    const duration_ns = finished_ns -| started_ns;
    if (duration_ns > policy.max_invoke_duration_ns) return error.InvokeDurationExceeded;

    return .{
        .result = result,
        .stdout = stdout,
        .stderr = stderr,
        .exit_code = resolver.exit_code,
        .memory_bytes = wasm_instance.memory_bytes.len,
        .duration_ns = duration_ns,
    };
}

fn checkWorkloadDeployable(
    allocator: std.mem.Allocator,
    path: []const u8,
    config: Config,
) !WorkloadCandidate {
    var loaded = try workload.Loaded.load(allocator, path);
    defer loaded.deinit(allocator);

    const requested_profile_name = loaded.manifest.targetName() orelse config.profile_name;
    const profile = try target_profile.resolve(requested_profile_name);

    try loaded.manifest.validateForTarget(profile);

    var network_report = try network.analyze(allocator, loaded.manifest.network, profile.network);
    defer network_report.deinit(allocator);
    if (!network_report.supported()) return error.WorkloadNetworkUnsupported;

    const wasm_bytes = try std.Io.Dir.cwd().readFileAlloc(
        std.Options.debug_io,
        loaded.wasm_path,
        allocator,
        .limited(scope.max_wasm_bytes),
    );
    defer allocator.free(wasm_bytes);

    var wasm_report = try wasm_compatibility.analyzeBytes(allocator, wasm_bytes, compatibilityTarget(profile), .{
        .initial_memory_bytes = loaded.manifest.requires.memory_bytes,
        .export_name = loaded.manifest.entrypointName(),
    });
    defer wasm_report.deinit(allocator);

    if (!wasm_report.supported()) return error.WorkloadWasmUnsupported;
    if (loaded.manifest.requires.wasi_nn and !hasMlImport(wasm_report)) return error.WorkloadMissingMlImport;

    if (loaded.model_path) |model_path| {
        var model_analysis = try analyzeOnnxPathDetailed(allocator, model_path);
        defer model_analysis.deinit(allocator);

        if (model_analysis.decode_error != null) return error.WorkloadModelDecodeFailed;
        if (model_analysis.byte_len > profile.max_model_bytes) return error.WorkloadModelTooLarge;
        if (!model_analysis.ok) return error.WorkloadModelUnsupported;
    } else if (loaded.manifest.requires.wasi_nn) {
        return error.WorkloadRequiresModel;
    }

    const owned_path = try allocator.dupe(u8, loaded.root_path);
    errdefer allocator.free(owned_path);

    const owned_name = try allocator.dupe(u8, loaded.manifest.name orelse loaded.root_path);
    errdefer allocator.free(owned_name);

    const owned_profile = try allocator.dupe(u8, profile.name);
    errdefer allocator.free(owned_profile);

    const owned_entrypoint = try allocator.dupe(u8, loaded.manifest.entrypointName());
    errdefer allocator.free(owned_entrypoint);

    const owned_wasm_path = try allocator.dupe(u8, loaded.wasm_path);
    errdefer allocator.free(owned_wasm_path);

    const model_path = loaded.model_path orelse "";
    const owned_model_path = try allocator.dupe(u8, model_path);
    errdefer allocator.free(owned_model_path);

    const memory_bytes = try workloadInitialMemoryBytes(loaded, profile);

    return .{
        .path = owned_path,
        .name = owned_name,
        .profile = owned_profile,
        .entrypoint = owned_entrypoint,
        .wasm_path = owned_wasm_path,
        .model_path = owned_model_path,
        .memory_bytes = memory_bytes,
    };
}

fn activityJson(allocator: std.mem.Allocator, state: State) ![]u8 {
    var out: std.Io.Writer.Allocating = .init(allocator);
    defer out.deinit();

    var json: std.json.Stringify = .{ .writer = &out.writer };
    try json.beginObject();
    try json.objectField("items");
    try json.beginArray();
    for (state.activities.items) |activity| {
        try json.beginObject();
        try json.objectField("id");
        try json.write(activity.id);
        try json.objectField("kind");
        try json.write(activityKindName(activity.kind));
        try json.objectField("status");
        try json.write(activityStatusName(activity.status));
        try json.objectField("path");
        try json.write(activity.path);
        try json.objectField("detail");
        try json.write(activity.detail);
        try json.endObject();
    }
    try json.endArray();
    try json.endObject();

    return try out.toOwnedSlice();
}

fn workloadsJson(allocator: std.mem.Allocator, state: State) ![]u8 {
    var out: std.Io.Writer.Allocating = .init(allocator);
    defer out.deinit();

    var json: std.json.Stringify = .{ .writer = &out.writer };
    try json.beginObject();
    try json.objectField("items");
    try json.beginArray();
    for (state.workloads.items) |registered| {
        try writeRegisteredWorkload(&json, registered);
    }
    try json.endArray();
    try json.endObject();

    return try out.toOwnedSlice();
}

fn telemetryJson(allocator: std.mem.Allocator, store: telemetry.Store) ![]u8 {
    var out: std.Io.Writer.Allocating = .init(allocator);
    defer out.deinit();

    var json: std.json.Stringify = .{ .writer = &out.writer };
    try json.beginObject();
    try json.objectField("events");
    try writeTelemetryEvents(&json, store);
    try json.objectField("metrics");
    try writeTelemetryMetrics(&json, store);
    try json.objectField("traces");
    try writeTelemetryTraces(&json, store);
    try json.endObject();

    return try out.toOwnedSlice();
}

fn telemetryEventsJson(allocator: std.mem.Allocator, store: telemetry.Store) ![]u8 {
    var out: std.Io.Writer.Allocating = .init(allocator);
    defer out.deinit();

    var json: std.json.Stringify = .{ .writer = &out.writer };
    try json.beginObject();
    try json.objectField("items");
    try writeTelemetryEvents(&json, store);
    try json.endObject();

    return try out.toOwnedSlice();
}

fn telemetryMetricsJson(allocator: std.mem.Allocator, store: telemetry.Store) ![]u8 {
    var out: std.Io.Writer.Allocating = .init(allocator);
    defer out.deinit();

    var json: std.json.Stringify = .{ .writer = &out.writer };
    try json.beginObject();
    try json.objectField("items");
    try writeTelemetryMetrics(&json, store);
    try json.endObject();

    return try out.toOwnedSlice();
}

fn telemetryTracesJson(allocator: std.mem.Allocator, store: telemetry.Store) ![]u8 {
    var out: std.Io.Writer.Allocating = .init(allocator);
    defer out.deinit();

    var json: std.json.Stringify = .{ .writer = &out.writer };
    try json.beginObject();
    try json.objectField("items");
    try writeTelemetryTraces(&json, store);
    try json.endObject();

    return try out.toOwnedSlice();
}

fn telemetryEventAcceptedJson(allocator: std.mem.Allocator, event: telemetry.Event) ![]u8 {
    var out: std.Io.Writer.Allocating = .init(allocator);
    defer out.deinit();

    var json: std.json.Stringify = .{ .writer = &out.writer };
    try json.beginObject();
    try json.objectField("kind");
    try json.write("telemetry_event");
    try json.objectField("accepted");
    try json.write(true);
    try json.objectField("event");
    try writeTelemetryEvent(&json, event);
    try json.endObject();

    return try out.toOwnedSlice();
}

fn telemetryMetricAcceptedJson(allocator: std.mem.Allocator, metric: telemetry.Metric) ![]u8 {
    var out: std.Io.Writer.Allocating = .init(allocator);
    defer out.deinit();

    var json: std.json.Stringify = .{ .writer = &out.writer };
    try json.beginObject();
    try json.objectField("kind");
    try json.write("telemetry_metric");
    try json.objectField("accepted");
    try json.write(true);
    try json.objectField("metric");
    try writeTelemetryMetric(&json, metric);
    try json.endObject();

    return try out.toOwnedSlice();
}

fn telemetryTraceAcceptedJson(allocator: std.mem.Allocator, span: telemetry.TraceSpan) ![]u8 {
    var out: std.Io.Writer.Allocating = .init(allocator);
    defer out.deinit();

    var json: std.json.Stringify = .{ .writer = &out.writer };
    try json.beginObject();
    try json.objectField("kind");
    try json.write("telemetry_trace");
    try json.objectField("accepted");
    try json.write(true);
    try json.objectField("span");
    try writeTelemetryTrace(&json, span);
    try json.endObject();

    return try out.toOwnedSlice();
}

fn otlpLogsJson(allocator: std.mem.Allocator, config: Config, store: telemetry.Store) ![]u8 {
    var out: std.Io.Writer.Allocating = .init(allocator);
    defer out.deinit();

    var json: std.json.Stringify = .{ .writer = &out.writer };
    try json.beginObject();
    try json.objectField("resourceLogs");
    try json.beginArray();
    try json.beginObject();
    try json.objectField("resource");
    try writeOtlpResource(&json, config);
    try json.objectField("scopeLogs");
    try json.beginArray();
    try json.beginObject();
    try json.objectField("scope");
    try writeOtlpScope(&json);
    try json.objectField("logRecords");
    try json.beginArray();
    for (store.events.items) |event| {
        try writeOtlpLogRecord(&json, event);
    }
    try json.endArray();
    try json.endObject();
    try json.endArray();
    try json.endObject();
    try json.endArray();
    try json.endObject();

    return try out.toOwnedSlice();
}

fn otlpTracesJson(allocator: std.mem.Allocator, config: Config, store: telemetry.Store) ![]u8 {
    var out: std.Io.Writer.Allocating = .init(allocator);
    defer out.deinit();

    var json: std.json.Stringify = .{ .writer = &out.writer };
    try json.beginObject();
    try json.objectField("resourceSpans");
    try json.beginArray();
    try json.beginObject();
    try json.objectField("resource");
    try writeOtlpResource(&json, config);
    try json.objectField("scopeSpans");
    try json.beginArray();
    try json.beginObject();
    try json.objectField("scope");
    try writeOtlpScope(&json);
    try json.objectField("spans");
    try json.beginArray();
    for (store.traces.items) |span| {
        try writeOtlpSpan(&json, span);
    }
    try json.endArray();
    try json.endObject();
    try json.endArray();
    try json.endObject();
    try json.endArray();
    try json.endObject();

    return try out.toOwnedSlice();
}

fn otlpMetricsJson(allocator: std.mem.Allocator, config: Config, store: telemetry.Store) ![]u8 {
    var out: std.Io.Writer.Allocating = .init(allocator);
    defer out.deinit();

    var json: std.json.Stringify = .{ .writer = &out.writer };
    try json.beginObject();
    try json.objectField("resourceMetrics");
    try json.beginArray();
    try json.beginObject();
    try json.objectField("resource");
    try writeOtlpResource(&json, config);
    try json.objectField("scopeMetrics");
    try json.beginArray();
    try json.beginObject();
    try json.objectField("scope");
    try writeOtlpScope(&json);
    try json.objectField("metrics");
    try json.beginArray();
    for (store.metrics.items) |metric| {
        try writeOtlpMetric(&json, metric);
    }
    try json.endArray();
    try json.endObject();
    try json.endArray();
    try json.endObject();
    try json.endArray();
    try json.endObject();

    return try out.toOwnedSlice();
}

fn writeOtlpResource(json: *std.json.Stringify, config: Config) !void {
    try json.beginObject();
    try json.objectField("attributes");
    try json.beginArray();
    try writeOtlpStringAttribute(json, "service.name", "zug");
    try writeOtlpStringAttribute(json, "service.namespace", "edge-runtime");
    try writeOtlpStringAttribute(json, "service.instance.id", config.node_id);
    try writeOtlpStringAttribute(json, "zug.profile", config.profile_name);
    try json.endArray();
    try json.endObject();
}

fn writeOtlpScope(json: *std.json.Stringify) !void {
    try json.beginObject();
    try json.objectField("name");
    try json.write("zug.agent");
    try json.objectField("version");
    try json.write(scope.runtime_scope_version);
    try json.endObject();
}

fn writeOtlpLogRecord(json: *std.json.Stringify, event: telemetry.Event) !void {
    try json.beginObject();
    try json.objectField("timeUnixNano");
    try writeU64JsonString(json, event.timestamp_ns);
    try json.objectField("severityText");
    try json.write(otlpSeverityText(event.severity));
    try json.objectField("body");
    try json.beginObject();
    try json.objectField("stringValue");
    try json.write(event.message);
    try json.endObject();
    try json.objectField("attributes");
    try json.beginArray();
    try writeOtlpStringAttribute(json, "zug.source", event.source);
    try writeOtlpStringAttribute(json, "zug.kind", event.kind);
    try writeOtlpStringAttribute(json, "event.name", event.kind);
    try json.endArray();
    try json.endObject();
}

fn writeOtlpSpan(json: *std.json.Stringify, span: telemetry.TraceSpan) !void {
    try json.beginObject();
    try json.objectField("traceId");
    try json.write(span.trace_id);
    try json.objectField("spanId");
    try json.write(span.span_id);
    if (span.parent_span_id.len != 0) {
        try json.objectField("parentSpanId");
        try json.write(span.parent_span_id);
    }
    try json.objectField("name");
    try json.write(span.name);
    try json.objectField("kind");
    try json.write(otlpSpanKindCode(span.kind));
    try json.objectField("startTimeUnixNano");
    try writeU64JsonString(json, span.start_time_ns);
    try json.objectField("endTimeUnixNano");
    try writeU64JsonString(json, span.end_time_ns);
    try json.objectField("status");
    try json.beginObject();
    try json.objectField("code");
    try json.write(otlpSpanStatusCode(span.status));
    try json.endObject();
    try json.objectField("attributes");
    try json.beginArray();
    try writeOtlpStringAttribute(json, "zug.source", span.source);
    try json.endArray();
    try json.endObject();
}

fn writeOtlpMetric(json: *std.json.Stringify, metric: telemetry.Metric) !void {
    try json.beginObject();
    try json.objectField("name");
    try json.write(metric.name);
    try json.objectField("unit");
    try json.write(metric.unit);
    try json.objectField("gauge");
    try json.beginObject();
    try json.objectField("dataPoints");
    try json.beginArray();
    try json.beginObject();
    try json.objectField("timeUnixNano");
    try writeU64JsonString(json, metric.timestamp_ns);
    try json.objectField("asDouble");
    try json.write(metric.value);
    try json.objectField("attributes");
    try json.beginArray();
    try writeOtlpStringAttribute(json, "zug.source", metric.source);
    try json.endArray();
    try json.endObject();
    try json.endArray();
    try json.endObject();
    try json.endObject();
}

fn writeOtlpStringAttribute(json: *std.json.Stringify, key: []const u8, value: []const u8) !void {
    try json.beginObject();
    try json.objectField("key");
    try json.write(key);
    try json.objectField("value");
    try json.beginObject();
    try json.objectField("stringValue");
    try json.write(value);
    try json.endObject();
    try json.endObject();
}

fn writeU64JsonString(json: *std.json.Stringify, value: u64) !void {
    var buffer: [20]u8 = undefined;
    const text = try std.fmt.bufPrint(&buffer, "{d}", .{value});
    try json.write(text);
}

fn otlpSeverityText(severity: telemetry.Severity) []const u8 {
    return switch (severity) {
        .info => "INFO",
        .warn => "WARN",
        .err => "ERROR",
    };
}

fn otlpSpanKindCode(kind: telemetry.SpanKind) u32 {
    return switch (kind) {
        .unspecified => 0,
        .internal => 1,
        .server => 2,
        .client => 3,
        .producer => 4,
        .consumer => 5,
    };
}

fn otlpSpanStatusCode(status: telemetry.SpanStatus) u32 {
    return switch (status) {
        .unset => 0,
        .ok => 1,
        .err => 2,
    };
}

fn writeTelemetryEvents(json: *std.json.Stringify, store: telemetry.Store) !void {
    try json.beginArray();
    for (store.events.items) |event| {
        try writeTelemetryEvent(json, event);
    }
    try json.endArray();
}

fn writeTelemetryMetrics(json: *std.json.Stringify, store: telemetry.Store) !void {
    try json.beginArray();
    for (store.metrics.items) |metric| {
        try writeTelemetryMetric(json, metric);
    }
    try json.endArray();
}

fn writeTelemetryTraces(json: *std.json.Stringify, store: telemetry.Store) !void {
    try json.beginArray();
    for (store.traces.items) |span| {
        try writeTelemetryTrace(json, span);
    }
    try json.endArray();
}

fn writeTelemetryEvent(json: *std.json.Stringify, event: telemetry.Event) !void {
    try json.beginObject();
    try json.objectField("id");
    try json.write(event.id);
    try json.objectField("timestamp_ns");
    try json.write(event.timestamp_ns);
    try json.objectField("source");
    try json.write(event.source);
    try json.objectField("kind");
    try json.write(event.kind);
    try json.objectField("severity");
    try json.write(telemetry.severityName(event.severity));
    try json.objectField("message");
    try json.write(event.message);
    try json.endObject();
}

fn writeTelemetryTrace(json: *std.json.Stringify, span: telemetry.TraceSpan) !void {
    try json.beginObject();
    try json.objectField("id");
    try json.write(span.id);
    try json.objectField("source");
    try json.write(span.source);
    try json.objectField("trace_id");
    try json.write(span.trace_id);
    try json.objectField("span_id");
    try json.write(span.span_id);
    try json.objectField("parent_span_id");
    try json.write(span.parent_span_id);
    try json.objectField("name");
    try json.write(span.name);
    try json.objectField("kind");
    try json.write(telemetry.spanKindName(span.kind));
    try json.objectField("start_time_ns");
    try json.write(span.start_time_ns);
    try json.objectField("end_time_ns");
    try json.write(span.end_time_ns);
    try json.objectField("status");
    try json.write(telemetry.spanStatusName(span.status));
    try json.endObject();
}

fn writeTelemetryMetric(json: *std.json.Stringify, metric: telemetry.Metric) !void {
    try json.beginObject();
    try json.objectField("id");
    try json.write(metric.id);
    try json.objectField("timestamp_ns");
    try json.write(metric.timestamp_ns);
    try json.objectField("source");
    try json.write(metric.source);
    try json.objectField("name");
    try json.write(metric.name);
    try json.objectField("value");
    try json.write(metric.value);
    try json.objectField("unit");
    try json.write(metric.unit);
    try json.endObject();
}

fn workloadLoadFailureJson(
    allocator: std.mem.Allocator,
    path: []const u8,
    err: anyerror,
    activity_id: u64,
) ![]u8 {
    var out: std.Io.Writer.Allocating = .init(allocator);
    defer out.deinit();

    var json: std.json.Stringify = .{ .writer = &out.writer };
    try json.beginObject();
    try json.objectField("kind");
    try json.write("workload_check");
    try json.objectField("path");
    try json.write(path);
    try json.objectField("activity_id");
    try json.write(activity_id);
    try json.objectField("supported");
    try json.write(false);
    try json.objectField("status");
    try json.write("fail");
    try json.objectField("load_error");
    try json.write(@errorName(err));
    try json.endObject();

    return try out.toOwnedSlice();
}

fn workloadProfileFailureJson(
    allocator: std.mem.Allocator,
    loaded: workload.Loaded,
    profile_name: []const u8,
    err: anyerror,
    activity_id: u64,
) ![]u8 {
    var out: std.Io.Writer.Allocating = .init(allocator);
    defer out.deinit();

    var json: std.json.Stringify = .{ .writer = &out.writer };
    try json.beginObject();
    try json.objectField("kind");
    try json.write("workload_check");
    try json.objectField("path");
    try json.write(loaded.root_path);
    try json.objectField("activity_id");
    try json.write(activity_id);
    try json.objectField("supported");
    try json.write(false);
    try json.objectField("status");
    try json.write("fail");
    try json.objectField("profile");
    try json.write(profile_name);
    try json.objectField("profile_error");
    try json.write(@errorName(err));
    try json.objectField("manifest");
    try writeWorkloadManifestSummary(&json, loaded, "");
    try json.endObject();

    return try out.toOwnedSlice();
}

fn workloadArtifactFailureJson(
    allocator: std.mem.Allocator,
    loaded: workload.Loaded,
    artifact: []const u8,
    err: anyerror,
    activity_id: u64,
) ![]u8 {
    var out: std.Io.Writer.Allocating = .init(allocator);
    defer out.deinit();

    var json: std.json.Stringify = .{ .writer = &out.writer };
    try json.beginObject();
    try json.objectField("kind");
    try json.write("workload_check");
    try json.objectField("path");
    try json.write(loaded.root_path);
    try json.objectField("activity_id");
    try json.write(activity_id);
    try json.objectField("supported");
    try json.write(false);
    try json.objectField("status");
    try json.write("fail");
    try json.objectField("artifact");
    try json.write(artifact);
    try json.objectField("artifact_error");
    try json.write(@errorName(err));
    try json.objectField("manifest");
    try writeWorkloadManifestSummary(&json, loaded, "");
    try json.endObject();

    return try out.toOwnedSlice();
}

fn workloadDeployFailureJson(
    allocator: std.mem.Allocator,
    path: []const u8,
    err: anyerror,
    activity_id: u64,
) ![]u8 {
    var out: std.Io.Writer.Allocating = .init(allocator);
    defer out.deinit();

    var json: std.json.Stringify = .{ .writer = &out.writer };
    try json.beginObject();
    try json.objectField("kind");
    try json.write("workload_deploy");
    try json.objectField("path");
    try json.write(path);
    try json.objectField("activity_id");
    try json.write(activity_id);
    try json.objectField("deployed");
    try json.write(false);
    try json.objectField("status");
    try json.write("rejected");
    try json.objectField("error");
    try json.write(@errorName(err));
    try json.endObject();

    return try out.toOwnedSlice();
}

fn workloadDeploySuccessJson(
    allocator: std.mem.Allocator,
    registered: RegisteredWorkload,
    activity_id: u64,
    existing: bool,
) ![]u8 {
    var out: std.Io.Writer.Allocating = .init(allocator);
    defer out.deinit();

    var json: std.json.Stringify = .{ .writer = &out.writer };
    try json.beginObject();
    try json.objectField("kind");
    try json.write("workload_deploy");
    try json.objectField("workload_id");
    try json.write(registered.id);
    try json.objectField("activity_id");
    try json.write(activity_id);
    try json.objectField("deployed");
    try json.write(true);
    try json.objectField("existing");
    try json.write(existing);
    try json.objectField("workload");
    try writeRegisteredWorkload(&json, registered);
    try json.endObject();

    return try out.toOwnedSlice();
}

fn writeRegisteredWorkload(json: *std.json.Stringify, registered: RegisteredWorkload) !void {
    try json.beginObject();
    try json.objectField("id");
    try json.write(registered.id);
    try json.objectField("status");
    try json.write(workloadStatusName(registered.status));
    try json.objectField("path");
    try json.write(registered.path);
    try json.objectField("name");
    try json.write(registered.name);
    try json.objectField("profile");
    try json.write(registered.profile);
    try json.objectField("entrypoint");
    try json.write(registered.entrypoint);
    try json.objectField("wasm");
    try json.write(registered.wasm_path);
    try json.objectField("model");
    try json.write(registered.model_path);
    try json.objectField("memory_bytes");
    try json.write(registered.memory_bytes);
    try json.objectField("last_error");
    try json.write(registered.last_error);
    try json.objectField("activity_id");
    try json.write(registered.activity_id);
    try json.endObject();
}

fn workloadInvokeMissingJson(allocator: std.mem.Allocator, workload_id: u64) ![]u8 {
    var out: std.Io.Writer.Allocating = .init(allocator);
    defer out.deinit();

    var json: std.json.Stringify = .{ .writer = &out.writer };
    try json.beginObject();
    try json.objectField("kind");
    try json.write("workload_invoke");
    try json.objectField("workload_id");
    try json.write(workload_id);
    try json.objectField("invoked");
    try json.write(false);
    try json.objectField("status");
    try json.write("missing");
    try json.objectField("error");
    try json.write("UnknownWorkload");
    try json.endObject();

    return try out.toOwnedSlice();
}

fn workloadInvokeFailureJson(
    allocator: std.mem.Allocator,
    registered: RegisteredWorkload,
    err: anyerror,
    activity_id: u64,
) ![]u8 {
    var out: std.Io.Writer.Allocating = .init(allocator);
    defer out.deinit();

    var json: std.json.Stringify = .{ .writer = &out.writer };
    try json.beginObject();
    try json.objectField("kind");
    try json.write("workload_invoke");
    try json.objectField("workload_id");
    try json.write(registered.id);
    try json.objectField("activity_id");
    try json.write(activity_id);
    try json.objectField("invoked");
    try json.write(false);
    try json.objectField("status");
    try json.write("failed");
    try json.objectField("error");
    try json.write(@errorName(err));
    try json.objectField("workload");
    try writeRegisteredWorkload(&json, registered);
    try json.endObject();

    return try out.toOwnedSlice();
}

fn workloadInvokeSuccessJson(
    allocator: std.mem.Allocator,
    registered: RegisteredWorkload,
    result: InvokeResult,
    activity_id: u64,
) ![]u8 {
    var out: std.Io.Writer.Allocating = .init(allocator);
    defer out.deinit();

    var json: std.json.Stringify = .{ .writer = &out.writer };
    try json.beginObject();
    try json.objectField("kind");
    try json.write("workload_invoke");
    try json.objectField("workload_id");
    try json.write(registered.id);
    try json.objectField("activity_id");
    try json.write(activity_id);
    try json.objectField("invoked");
    try json.write(true);
    try json.objectField("status");
    try json.write("completed");
    try json.objectField("duration_ns");
    try json.write(result.duration_ns);
    try json.objectField("memory_bytes");
    try json.write(result.memory_bytes);
    try json.objectField("exit_code");
    try json.write(result.exit_code orelse 0);
    try json.objectField("exit_code_present");
    try json.write(result.exit_code != null);
    try json.objectField("stdout");
    try json.write(result.stdout);
    try json.objectField("stderr");
    try json.write(result.stderr);
    try json.objectField("result");
    try writeWasmValue(&json, result.result);
    try json.objectField("workload");
    try writeRegisteredWorkload(&json, registered);
    try json.endObject();

    return try out.toOwnedSlice();
}

fn writeWasmValue(json: *std.json.Stringify, value: ?wasm_interpreter.Value) !void {
    try json.beginObject();
    if (value) |actual| {
        switch (actual) {
            .i32 => |payload| {
                try json.objectField("type");
                try json.write("i32");
                try json.objectField("value");
                try json.write(@as(i32, @bitCast(payload)));
            },
            .i64 => |payload| {
                try json.objectField("type");
                try json.write("i64");
                try json.objectField("value");
                try json.write(@as(i64, @bitCast(payload)));
            },
            .f32 => |payload| {
                try json.objectField("type");
                try json.write("f32");
                try json.objectField("value");
                try json.write(payload);
            },
            .f64 => |payload| {
                try json.objectField("type");
                try json.write("f64");
                try json.objectField("value");
                try json.write(payload);
            },
            .funcref => |payload| {
                try json.objectField("type");
                try json.write("funcref");
                try json.objectField("value");
                try json.write(payload orelse 0);
                try json.objectField("null");
                try json.write(payload == null);
            },
            .v128 => |payload| {
                try json.objectField("type");
                try json.write("v128");
                try json.objectField("bytes");
                try json.beginArray();
                for (payload) |byte| {
                    try json.write(byte);
                }
                try json.endArray();
            },
        }
    } else {
        try json.objectField("type");
        try json.write("none");
    }
    try json.endObject();
}

fn workloadCheckReportJson(
    allocator: std.mem.Allocator,
    loaded: workload.Loaded,
    profile: target_profile.Profile,
    manifest_error: ?[]const u8,
    network_report: network.Report,
    wasm_byte_len: usize,
    wasm_report: wasm_compatibility.Report,
    missing_ml_import: bool,
    model_analysis: ?OnnxAnalysis,
    model_size_error: bool,
    ok: bool,
    activity_id: u64,
) ![]u8 {
    var out: std.Io.Writer.Allocating = .init(allocator);
    defer out.deinit();

    var json: std.json.Stringify = .{ .writer = &out.writer };
    try json.beginObject();
    try json.objectField("kind");
    try json.write("workload_check");
    try json.objectField("path");
    try json.write(loaded.root_path);
    try json.objectField("activity_id");
    try json.write(activity_id);
    try json.objectField("supported");
    try json.write(ok);
    try json.objectField("status");
    try json.write(if (ok) "pass" else "fail");
    try json.objectField("profile");
    try json.write(profile.name);

    try json.objectField("manifest");
    try writeWorkloadManifestSummary(&json, loaded, manifest_error orelse "");

    try json.objectField("checks");
    try json.beginObject();
    try json.objectField("missing_ml_import");
    try json.write(missing_ml_import);
    try json.endObject();

    try json.objectField("network");
    try writeNetworkReport(&json, network_report);

    try json.objectField("wasm");
    try writeWasmReport(&json, loaded.wasm_path, wasm_byte_len, wasm_report);

    try json.objectField("model");
    try writeModelAnalysis(&json, model_analysis, model_size_error);

    try json.endObject();

    return try out.toOwnedSlice();
}

fn writeWorkloadManifestSummary(
    json: *std.json.Stringify,
    loaded: workload.Loaded,
    manifest_error: []const u8,
) !void {
    try json.beginObject();
    try json.objectField("path");
    try json.write(loaded.manifest_path);
    try json.objectField("name");
    try writeOptionalStringAsEmpty(json, loaded.manifest.name);
    try json.objectField("entrypoint");
    try json.write(loaded.manifest.entrypointName());
    try json.objectField("wasm");
    try json.write(loaded.wasm_path);
    try json.objectField("model");
    try writeOptionalStringAsEmpty(json, loaded.model_path);
    try json.objectField("error");
    try json.write(manifest_error);
    try json.endObject();
}

fn writeNetworkReport(json: *std.json.Stringify, report: network.Report) !void {
    try json.beginObject();
    try json.objectField("supported");
    try json.write(report.supported());
    try json.objectField("unsupported_count");
    try json.write(report.unsupported_count);

    try json.objectField("protocols");
    try json.beginArray();
    for (report.protocol_checks.items) |check| {
        try json.beginObject();
        try json.objectField("subject");
        try json.write(check.subject);
        try json.objectField("protocol");
        try json.write(network.protocolName(check.protocol));
        try json.objectField("supported");
        try json.write(check.supported);
        try json.endObject();
    }
    try json.endArray();

    try json.objectField("max_inflight");
    try json.beginObject();
    try json.objectField("specified");
    try json.write(report.requested_max_inflight != null);
    try json.objectField("requested");
    try json.write(report.requested_max_inflight orelse 0);
    try json.objectField("target");
    try json.write(report.target_max_inflight);
    try json.objectField("supported");
    try json.write(report.max_inflight_supported);
    try json.endObject();

    try json.objectField("request_timeout_ms");
    try json.beginObject();
    try json.objectField("specified");
    try json.write(report.requested_timeout_ms != null);
    try json.objectField("requested");
    try json.write(report.requested_timeout_ms orelse 0);
    try json.objectField("target");
    try json.write(report.target_max_timeout_ms);
    try json.objectField("supported");
    try json.write(report.timeout_supported);
    try json.endObject();

    try json.objectField("public_egress");
    try json.beginObject();
    try json.objectField("requested");
    try json.write(report.public_egress_requested);
    try json.objectField("supported");
    try json.write(report.public_egress_supported);
    try json.endObject();

    try json.endObject();
}

fn writeWasmReport(
    json: *std.json.Stringify,
    path: []const u8,
    byte_len: usize,
    report: wasm_compatibility.Report,
) !void {
    try json.beginObject();
    try json.objectField("path");
    try json.write(path);
    try json.objectField("bytes");
    try json.write(byte_len);
    try json.objectField("supported");
    try json.write(report.supported());
    try json.objectField("parse_error");
    try writeOptionalErrorAsEmpty(json, report.parse_error);
    try json.objectField("validate_error");
    try writeOptionalErrorAsEmpty(json, report.validate_error);
    try json.objectField("unsupported_sections");
    try json.write(report.sections.unsupported_count);
    try json.objectField("unsupported_imports");
    try json.write(report.imports.unsupported_function_imports + report.imports.unsupported_non_function_imports);
    try json.objectField("unsupported_wasi_requirements");
    try json.write(report.wasi_requirements.unsupported_count);
    try json.objectField("unsupported_opcodes");
    try json.write(report.opcodes.unsupported.items.len);
    try json.objectField("unsupported_memory_features");
    try json.write(report.memory_encoding.unsupported_flags + report.memory_encoding.shared + report.memory_encoding.memory64);
    try json.objectField("requested_export");
    try writeOptionalStringAsEmpty(json, report.exports.requested_export);
    try json.objectField("has_requested_export");
    try json.write(report.exports.has_requested_export);
    try json.endObject();
}

fn writeModelAnalysis(
    json: *std.json.Stringify,
    model_analysis: ?OnnxAnalysis,
    model_size_error: bool,
) !void {
    try json.beginObject();

    if (model_analysis) |analysis| {
        try json.objectField("present");
        try json.write(true);
        try json.objectField("path");
        try json.write(analysis.path);
        try json.objectField("bytes");
        try json.write(analysis.byte_len);
        try json.objectField("read_error");
        try writeOptionalStringAsEmpty(json, analysis.read_error);
        try json.objectField("decode_error");
        try writeOptionalStringAsEmpty(json, analysis.decode_error);
        try json.objectField("model_size_error");
        try json.write(model_size_error);
        try json.objectField("supported");
        try json.write(analysis.ok and !model_size_error);

        try json.objectField("operators");
        try json.beginArray();
        if (analysis.report) |report| {
            for (report.operators.items) |operator| {
                try json.beginObject();
                try json.objectField("domain");
                try json.write(operator.domain);
                try json.objectField("op_type");
                try json.write(operator.op_type);
                try json.objectField("count");
                try json.write(operator.count);
                try json.objectField("supported");
                try json.write(operator.supported);
                try json.endObject();
            }
        }
        try json.endArray();

        try json.objectField("issues");
        try json.beginArray();
        if (analysis.report) |report| {
            for (report.issues.items) |issue| {
                try json.beginObject();
                try json.objectField("severity");
                try json.write(onnxIssueSeverity(issue.kind));
                try json.objectField("kind");
                try json.write(@tagName(issue.kind));
                try json.objectField("subject");
                try json.write(issue.subject);
                try json.objectField("detail");
                try json.write(issue.detail);
                try json.endObject();
            }
        }
        try json.endArray();
    } else {
        try json.objectField("present");
        try json.write(false);
        try json.objectField("supported");
        try json.write(true);
    }

    try json.endObject();
}

fn analyzeOnnxPathDetailed(allocator: std.mem.Allocator, path: []const u8) !OnnxAnalysis {
    const bytes = try std.Io.Dir.cwd().readFileAlloc(
        std.Options.debug_io,
        path,
        allocator,
        .limited(scope.max_model_bytes),
    );
    defer allocator.free(bytes);

    var analysis = OnnxAnalysis{
        .path = path,
        .byte_len = bytes.len,
    };
    errdefer analysis.deinit(allocator);

    var reader: std.Io.Reader = .fixed(bytes);
    analysis.model = onnx.ModelProto.decode(&reader, allocator) catch |err| {
        analysis.decode_error = @errorName(err);
        return analysis;
    };

    analysis.report = try capabilities.analyze(allocator, &analysis.model.?);
    analysis.ok = onnxReportPassesCheck(analysis.report.?);

    return analysis;
}

fn onnxReportPassesCheck(report: capabilities.Report) bool {
    for (report.issues.items) |issue| {
        if (onnxIssueIsFatal(issue.kind)) return false;
    }

    return true;
}

fn onnxIssueSeverity(kind: capabilities.IssueKind) []const u8 {
    return if (onnxIssueIsFatal(kind)) "error" else "warning";
}

fn onnxIssueIsFatal(kind: capabilities.IssueKind) bool {
    return switch (kind) {
        .dynamic_dimension => false,
        else => true,
    };
}

fn compatibilityTarget(profile: target_profile.Profile) wasm_compatibility.Target {
    return .{
        .default_memory_bytes = profile.default_memory_bytes,
        .max_memory_bytes = profile.max_memory_bytes,
    };
}

fn workloadInitialMemoryBytes(loaded: workload.Loaded, profile: target_profile.Profile) !usize {
    const requested = loaded.manifest.requires.memory_bytes orelse profile.default_memory_bytes;
    const aligned = try alignToWasmPage(requested);
    if (aligned > profile.max_memory_bytes) return error.WorkloadMemoryTooLarge;
    return aligned;
}

fn alignToWasmPage(value: usize) !usize {
    const wasm_page_size: usize = 64 * 1024;
    const plus_page = try std.math.add(usize, value, wasm_page_size - 1);
    return plus_page - (plus_page % wasm_page_size);
}

fn monotonicNowNs() u64 {
    const value = std.Io.Clock.awake.now(std.Options.debug_io).toNanoseconds();
    return std.math.cast(u64, value) orelse 0;
}

fn hasMlImport(report: wasm_compatibility.Report) bool {
    for (report.imports.details.items) |import| {
        if (std.mem.eql(u8, import.module_name, "wasi_nn") or
            std.mem.eql(u8, import.module_name, "zug_nn"))
        {
            return true;
        }
    }

    return false;
}

fn writeOptionalStringAsEmpty(json: *std.json.Stringify, value: ?[]const u8) !void {
    if (value) |actual| {
        try json.write(actual);
    } else {
        try json.write("");
    }
}

fn writeOptionalErrorAsEmpty(json: *std.json.Stringify, value: ?anyerror) !void {
    if (value) |actual| {
        try json.write(@errorName(actual));
    } else {
        try json.write("");
    }
}

fn activityKindName(kind: ActivityKind) []const u8 {
    return switch (kind) {
        .workload_check => "workload_check",
        .workload_deploy => "workload_deploy",
        .workload_invoke => "workload_invoke",
    };
}

fn activityStatusName(status: ActivityStatus) []const u8 {
    return switch (status) {
        .accepted => "accepted",
        .rejected => "rejected",
        .failed => "failed",
    };
}

fn workloadStatusName(status: WorkloadStatus) []const u8 {
    return switch (status) {
        .accepted => "accepted",
        .loaded => "loaded",
        .failed => "failed",
    };
}

pub fn capabilitiesJson(allocator: std.mem.Allocator, config: Config) ![]u8 {
    const profile = try target_profile.resolve(config.profile_name);
    var out: std.Io.Writer.Allocating = .init(allocator);
    defer out.deinit();

    var json: std.json.Stringify = .{ .writer = &out.writer };
    try json.beginObject();

    try json.objectField("runtime");
    try json.beginObject();
    try json.objectField("name");
    try json.write("zug");
    try json.objectField("agent_protocol");
    try json.write("zug-agent/v0");
    try json.endObject();

    try json.objectField("node_id");
    try json.write(config.node_id);

    try json.objectField("profile");
    try json.write(profile.name);

    try json.objectField("resources");
    try json.beginObject();
    try json.objectField("default_memory_bytes");
    try json.write(profile.default_memory_bytes);
    try json.objectField("max_memory_bytes");
    try json.write(profile.max_memory_bytes);
    try json.objectField("max_model_bytes");
    try json.write(profile.max_model_bytes);
    try json.endObject();

    try json.objectField("wasi");
    try json.beginObject();
    try json.objectField("wasi_nn");
    try json.write(profile.supports_wasi_nn);
    try json.objectField("wasi_http");
    try json.write(profile.supports_wasi_http);
    try json.endObject();

    try json.objectField("dtypes");
    try json.beginArray();
    for (profile.supported_dtypes) |dtype| {
        try json.write(dtypeName(dtype));
    }
    try json.endArray();

    try json.objectField("network");
    try writeNetworkCapabilities(&json, profile.network);

    try json.objectField("gpu");
    try writeGpuCapabilities(&json, config.gpu);

    try json.objectField("accelerators");
    try writeAcceleratorCapabilities(&json, config.accelerators);

    try json.objectField("telemetry");
    try writeTelemetryCapabilities(&json);

    try json.objectField("policy");
    try writeRuntimePolicy(&json, config.policy);

    try json.endObject();
    return try out.toOwnedSlice();
}

fn writeGpuCapabilities(json: *std.json.Stringify, caps: gpu.Capabilities) !void {
    try json.beginObject();
    try json.objectField("enabled");
    try json.write(caps.enabled);
    try json.objectField("device_count");
    try json.write(caps.deviceCount());

    try json.objectField("devices");
    try json.beginArray();
    if (caps.enabled) {
        for (caps.devices, 0..) |device, index| {
            try json.beginObject();
            try json.objectField("index");
            try json.write(index);
            try json.objectField("kind");
            try json.write(gpu.deviceKindName(device.kind));
            try json.objectField("backend");
            try json.write(accelerator.backendKindName(device.backend));
            try json.objectField("total_memory_bytes");
            try json.write(device.total_memory_bytes);
            try json.objectField("available_memory_bytes");
            try json.write(device.available_memory_bytes);
            try json.objectField("queue_count");
            try json.write(device.queue_count);
            try json.endObject();
        }
    }
    try json.endArray();

    try json.endObject();
}

fn writeAcceleratorCapabilities(json: *std.json.Stringify, caps: accelerator.Capabilities) !void {
    try json.beginObject();
    try json.objectField("backends");
    try json.beginArray();
    for (caps.backends) |backend| {
        try json.beginObject();
        try json.objectField("kind");
        try json.write(accelerator.backendKindName(backend.kind));
        try json.objectField("status");
        try json.write(accelerator.backendStatusName(backend.status));
        try json.objectField("device_count");
        try json.write(backend.device_count);
        try json.objectField("total_memory_bytes");
        try json.write(backend.total_memory_bytes);
        try json.objectField("available_memory_bytes");
        try json.write(backend.available_memory_bytes);
        try json.objectField("graph_execution");
        try json.write(backend.supportsGraphExecution());
        try json.objectField("features");
        try json.beginArray();
        for (backend.features) |feature| {
            try json.write(accelerator.featureName(feature));
        }
        try json.endArray();
        try json.endObject();
    }
    try json.endArray();

    try json.objectField("default_gpu_backend");
    if (caps.defaultGpuBackend()) |backend| {
        try json.write(accelerator.backendKindName(backend));
    } else {
        try json.write("");
    }
    try json.endObject();
}

fn writeRuntimePolicy(json: *std.json.Stringify, policy: RuntimePolicy) !void {
    try json.beginObject();
    try json.objectField("max_request_body_bytes");
    try json.write(policy.max_request_body_bytes);
    try json.objectField("max_invoke_args");
    try json.write(policy.max_invoke_args);
    try json.objectField("max_stdin_bytes");
    try json.write(policy.max_stdin_bytes);
    try json.objectField("max_stdout_bytes");
    try json.write(policy.max_stdout_bytes);
    try json.objectField("max_stderr_bytes");
    try json.write(policy.max_stderr_bytes);
    try json.objectField("max_invoke_duration_ns");
    try json.write(policy.max_invoke_duration_ns);
    try json.endObject();
}

fn writeTelemetryCapabilities(json: *std.json.Stringify) !void {
    try json.beginObject();
    try json.objectField("events");
    try json.write(true);
    try json.objectField("metrics");
    try json.write(true);
    try json.objectField("traces");
    try json.write(true);
    try json.objectField("storage");
    try json.write("memory");
    try json.objectField("opentelemetry");
    try json.beginObject();
    try json.objectField("otlp_http_json");
    try json.write(true);
    try json.objectField("logs_path");
    try json.write("/v1/logs");
    try json.objectField("metrics_path");
    try json.write("/v1/metrics");
    try json.objectField("traces_path");
    try json.write("/v1/traces");
    try json.endObject();
    try json.objectField("endpoints");
    try json.beginArray();
    try json.write("GET /telemetry");
    try json.write("GET /telemetry/events");
    try json.write("GET /telemetry/metrics");
    try json.write("GET /telemetry/traces");
    try json.write("GET /telemetry/otlp/logs");
    try json.write("GET /telemetry/otlp/metrics");
    try json.write("GET /telemetry/otlp/traces");
    try json.write("POST /telemetry/events");
    try json.write("POST /telemetry/metrics");
    try json.write("POST /telemetry/traces");
    try json.write("POST /v1/logs");
    try json.write("POST /v1/metrics");
    try json.write("POST /v1/traces");
    try json.endArray();
    try json.endObject();
}

fn healthJson(allocator: std.mem.Allocator, config: Config) ![]u8 {
    var out: std.Io.Writer.Allocating = .init(allocator);
    defer out.deinit();

    var json: std.json.Stringify = .{ .writer = &out.writer };
    try json.beginObject();
    try json.objectField("status");
    try json.write("ok");
    try json.objectField("node_id");
    try json.write(config.node_id);
    try json.objectField("profile");
    try json.write(config.profile_name);
    try json.endObject();

    return try out.toOwnedSlice();
}

fn writeNetworkCapabilities(json: *std.json.Stringify, caps: network.Capabilities) !void {
    try json.beginObject();

    try json.objectField("protocols");
    try json.beginArray();
    for (caps.protocols) |protocol| {
        try json.write(network.protocolName(protocol));
    }
    try json.endArray();

    try json.objectField("max_inflight");
    try json.write(caps.max_inflight);
    try json.objectField("max_request_timeout_ms");
    try json.write(caps.max_request_timeout_ms);
    try json.objectField("public_egress");
    try json.write(caps.public_egress);

    try json.endObject();
}

fn jsonResponse(allocator: std.mem.Allocator, status: Status, body: []const u8) ![]u8 {
    var out: std.Io.Writer.Allocating = .init(allocator);
    defer out.deinit();

    try out.writer.print(
        "HTTP/1.1 {d} {s}\r\ncontent-type: application/json\r\ncontent-length: {d}\r\nconnection: close\r\n\r\n",
        .{ status.code, status.reason, body.len },
    );
    try out.writer.writeAll(body);

    return try out.toOwnedSlice();
}

fn parseRequestLine(request: []const u8) !RequestLine {
    const line_end = std.mem.indexOfScalar(u8, request, '\n') orelse return error.InvalidHttpRequest;
    const line = std.mem.trim(u8, request[0..line_end], "\r");

    var parts = std.mem.splitScalar(u8, line, ' ');
    const method = parts.next() orelse return error.InvalidHttpRequest;
    const target = parts.next() orelse return error.InvalidHttpRequest;
    const version = parts.next() orelse return error.InvalidHttpRequest;

    if (method.len == 0 or target.len == 0 or
        !std.mem.startsWith(u8, version, "HTTP/"))
    {
        return error.InvalidHttpRequest;
    }

    return .{
        .method = method,
        .target = target,
    };
}

fn requestPath(target: []const u8) []const u8 {
    const query_index = std.mem.indexOfScalar(u8, target, '?') orelse return target;
    return target[0..query_index];
}

pub fn parseListenAddress(text: []const u8) !net.IpAddress {
    if (net.IpAddress.parseLiteral(text)) |address| {
        return address;
    } else |parse_error| {
        if (text.len == 0 or text[0] == '[') return parse_error;

        const colon_index = std.mem.lastIndexOfScalar(u8, text, ':') orelse return parse_error;
        const host = text[0..colon_index];
        const port = std.fmt.parseInt(u16, text[colon_index + 1 ..], 10) catch return error.InvalidPort;

        if (std.mem.eql(u8, host, "localhost")) {
            return .{ .ip4 = .loopback(port) };
        }

        if (std.mem.eql(u8, host, "*")) {
            return .{ .ip4 = .unspecified(port) };
        }

        return parse_error;
    }
}

fn dtypeName(dtype: target_profile.DType) []const u8 {
    return switch (dtype) {
        .float32 => "float32",
        .int64 => "int64",
        .int32 => "int32",
        .uint8 => "uint8",
        .bool => "bool",
    };
}

test "agent capabilities JSON exposes profile resources and networking" {
    const gpu_devices = [_]gpu.Device{.{
        .kind = .integrated,
        .total_memory_bytes = 1024,
        .available_memory_bytes = 512,
        .queue_count = 1,
    }};

    const body = try capabilitiesJson(std.testing.allocator, .{
        .node_id = "edge-a",
        .profile_name = "vision-f32-basic",
        .gpu = .{
            .enabled = true,
            .devices = &gpu_devices,
        },
    });
    defer std.testing.allocator.free(body);

    try std.testing.expect(std.mem.indexOf(u8, body, "\"node_id\":\"edge-a\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, body, "\"profile\":\"vision-f32-basic\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, body, "\"http\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, body, "\"tcp\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, body, "\"policy\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, body, "\"max_invoke_args\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, body, "\"gpu\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, body, "\"device_count\":1") != null);
    try std.testing.expect(std.mem.indexOf(u8, body, "\"kind\":\"integrated\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, body, "\"accelerators\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, body, "\"kind\":\"cuda\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, body, "\"status\":\"planned\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, body, "\"telemetry\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, body, "\"POST /telemetry/events\"") != null);
}

test "agent HTTP responses route health capabilities and not found" {
    const health = try responseForRequest(
        std.testing.allocator,
        "GET /health HTTP/1.1\r\nhost: local\r\n\r\n",
        .{ .node_id = "edge-a" },
    );
    defer std.testing.allocator.free(health);

    try std.testing.expect(std.mem.indexOf(u8, health, "HTTP/1.1 200 OK") != null);
    try std.testing.expect(std.mem.indexOf(u8, health, "\"status\":\"ok\"") != null);

    const capabilities_response = try responseForRequest(
        std.testing.allocator,
        "GET /capabilities HTTP/1.1\r\nhost: local\r\n\r\n",
        .{ .node_id = "edge-a" },
    );
    defer std.testing.allocator.free(capabilities_response);
    try std.testing.expect(std.mem.indexOf(u8, capabilities_response, "\"agent_protocol\":\"zug-agent/v0\"") != null);

    const missing = try responseForRequest(
        std.testing.allocator,
        "GET /missing HTTP/1.1\r\nhost: local\r\n\r\n",
        .{},
    );
    defer std.testing.allocator.free(missing);
    try std.testing.expect(std.mem.indexOf(u8, missing, "HTTP/1.1 404 Not Found") != null);
}

test "agent workload check reports failure and records activity" {
    var state = State{};
    defer state.deinit(std.testing.allocator);

    const response = try responseForRequestWithState(
        std.testing.allocator,
        "POST /workloads/check HTTP/1.1\r\nhost: local\r\n\r\n{\"path\":\"missing-workload\"}",
        .{},
        &state,
    );
    defer std.testing.allocator.free(response);

    try std.testing.expect(std.mem.indexOf(u8, response, "HTTP/1.1 200 OK") != null);
    try std.testing.expect(std.mem.indexOf(u8, response, "\"kind\":\"workload_check\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, response, "\"supported\":false") != null);
    try std.testing.expect(std.mem.indexOf(u8, response, "\"load_error\"") != null);
    try std.testing.expectEqual(@as(usize, 1), state.activities.items.len);
    try std.testing.expectEqual(ActivityStatus.rejected, state.activities.items[0].status);
    try std.testing.expectEqualStrings("missing-workload", state.activities.items[0].path);

    const activity = try responseForRequestWithState(
        std.testing.allocator,
        "GET /activity HTTP/1.1\r\nhost: local\r\n\r\n",
        .{},
        &state,
    );
    defer std.testing.allocator.free(activity);

    try std.testing.expect(std.mem.indexOf(u8, activity, "\"status\":\"rejected\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, activity, "\"path\":\"missing-workload\"") != null);
}

test "agent workload deploy rejects unsupported path" {
    var state = State{};
    defer state.deinit(std.testing.allocator);

    const response = try responseForRequestWithState(
        std.testing.allocator,
        "POST /workloads/deploy HTTP/1.1\r\nhost: local\r\n\r\n{\"path\":\"missing-workload\"}",
        .{},
        &state,
    );
    defer std.testing.allocator.free(response);

    try std.testing.expect(std.mem.indexOf(u8, response, "HTTP/1.1 200 OK") != null);
    try std.testing.expect(std.mem.indexOf(u8, response, "\"kind\":\"workload_deploy\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, response, "\"deployed\":false") != null);
    try std.testing.expectEqual(@as(usize, 0), state.workloads.items.len);
    try std.testing.expectEqual(@as(usize, 1), state.activities.items.len);
    try std.testing.expectEqual(ActivityKind.workload_deploy, state.activities.items[0].kind);
    try std.testing.expectEqual(ActivityStatus.rejected, state.activities.items[0].status);
}

test "agent workload registry lists registered workloads" {
    var state = State{};
    defer state.deinit(std.testing.allocator);

    const activity_id = try state.record(std.testing.allocator, .workload_deploy, .accepted, "workloads/demo", "accepted");
    const workload_id = try state.registerWorkload(std.testing.allocator, .{
        .path = "workloads/demo",
        .name = "demo",
        .profile = "vision-f32-basic",
        .entrypoint = "run",
        .wasm_path = "workloads/demo/guest.wasm",
        .model_path = "workloads/demo/model.onnx",
        .memory_bytes = 2 * 1024 * 1024,
    }, activity_id);

    const response = try responseForRequestWithState(
        std.testing.allocator,
        "GET /workloads HTTP/1.1\r\nhost: local\r\n\r\n",
        .{},
        &state,
    );
    defer std.testing.allocator.free(response);

    try std.testing.expectEqual(@as(u64, 1), workload_id);
    try std.testing.expect(std.mem.indexOf(u8, response, "\"id\":1") != null);
    try std.testing.expect(std.mem.indexOf(u8, response, "\"status\":\"accepted\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, response, "\"path\":\"workloads/demo\"") != null);
}

test "agent records telemetry events and metrics" {
    var state = State{};
    defer state.deinit(std.testing.allocator);

    const event_response = try responseForRequestWithState(
        std.testing.allocator,
        "POST /telemetry/events HTTP/1.1\r\nhost: local\r\n\r\n{\"source\":\"camera\",\"kind\":\"frame_drop\",\"severity\":\"warn\",\"message\":\"dropped frame\"}",
        .{},
        &state,
    );
    defer std.testing.allocator.free(event_response);

    try std.testing.expect(std.mem.indexOf(u8, event_response, "\"kind\":\"telemetry_event\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, event_response, "\"accepted\":true") != null);
    try std.testing.expectEqual(@as(usize, 1), state.telemetry_store.events.items.len);
    try std.testing.expectEqualStrings("frame_drop", state.telemetry_store.events.items[0].kind);
    try std.testing.expectEqual(telemetry.Severity.warn, state.telemetry_store.events.items[0].severity);

    const metric_response = try responseForRequestWithState(
        std.testing.allocator,
        "POST /telemetry/metrics HTTP/1.1\r\nhost: local\r\n\r\n{\"source\":\"runtime\",\"name\":\"latency_ms\",\"value\":14.5,\"unit\":\"ms\"}",
        .{},
        &state,
    );
    defer std.testing.allocator.free(metric_response);

    try std.testing.expect(std.mem.indexOf(u8, metric_response, "\"kind\":\"telemetry_metric\"") != null);
    try std.testing.expectEqual(@as(usize, 1), state.telemetry_store.metrics.items.len);
    try std.testing.expectEqualStrings("latency_ms", state.telemetry_store.metrics.items[0].name);
    try std.testing.expectEqual(@as(f64, 14.5), state.telemetry_store.metrics.items[0].value);

    const telemetry_response = try responseForRequestWithState(
        std.testing.allocator,
        "GET /telemetry HTTP/1.1\r\nhost: local\r\n\r\n",
        .{},
        &state,
    );
    defer std.testing.allocator.free(telemetry_response);

    try std.testing.expect(std.mem.indexOf(u8, telemetry_response, "\"events\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, telemetry_response, "\"metrics\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, telemetry_response, "\"frame_drop\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, telemetry_response, "\"latency_ms\"") != null);
}

test "agent records local telemetry traces" {
    var state = State{};
    defer state.deinit(std.testing.allocator);

    const response = try responseForRequestWithState(
        std.testing.allocator,
        "POST /telemetry/traces HTTP/1.1\r\nhost: local\r\n\r\n{\"source\":\"runtime\",\"trace_id\":\"5b8efff798038103d269b633813fc60c\",\"span_id\":\"eee19b7ec3c1b174\",\"name\":\"invoke\",\"kind\":\"server\",\"start_time_ns\":100,\"end_time_ns\":150,\"status\":\"ok\"}",
        .{},
        &state,
    );
    defer std.testing.allocator.free(response);

    try std.testing.expect(std.mem.indexOf(u8, response, "\"kind\":\"telemetry_trace\"") != null);
    try std.testing.expectEqual(@as(usize, 1), state.telemetry_store.traces.items.len);
    try std.testing.expectEqualStrings("invoke", state.telemetry_store.traces.items[0].name);
    try std.testing.expectEqual(telemetry.SpanKind.server, state.telemetry_store.traces.items[0].kind);
    try std.testing.expectEqual(telemetry.SpanStatus.ok, state.telemetry_store.traces.items[0].status);

    const traces_response = try responseForRequestWithState(
        std.testing.allocator,
        "GET /telemetry/traces HTTP/1.1\r\nhost: local\r\n\r\n",
        .{},
        &state,
    );
    defer std.testing.allocator.free(traces_response);

    try std.testing.expect(std.mem.indexOf(u8, traces_response, "\"trace_id\":\"5b8efff798038103d269b633813fc60c\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, traces_response, "\"kind\":\"server\"") != null);
}

test "agent ingests and exports OTLP HTTP JSON telemetry" {
    var state = State{};
    defer state.deinit(std.testing.allocator);

    const logs_request =
        \\POST /v1/logs HTTP/1.1
        \\host: local
        \\
        \\{"resourceLogs":[{"resource":{"attributes":[{"key":"service.name","value":{"stringValue":"edge-camera"}}]},"scopeLogs":[{"scope":{"name":"demo"},"logRecords":[{"timeUnixNano":"123","severityText":"WARN","body":{"stringValue":"frame dropped"},"attributes":[{"key":"event.name","value":{"stringValue":"frame_drop"}}]}]}]}]}
    ;
    const logs_response = try responseForRequestWithState(
        std.testing.allocator,
        logs_request,
        .{},
        &state,
    );
    defer std.testing.allocator.free(logs_response);

    try std.testing.expect(std.mem.indexOf(u8, logs_response, "HTTP/1.1 200 OK") != null);
    try std.testing.expectEqual(@as(usize, 1), state.telemetry_store.events.items.len);
    try std.testing.expectEqualStrings("edge-camera", state.telemetry_store.events.items[0].source);
    try std.testing.expectEqualStrings("frame_drop", state.telemetry_store.events.items[0].kind);
    try std.testing.expectEqual(telemetry.Severity.warn, state.telemetry_store.events.items[0].severity);

    const metrics_request =
        \\POST /v1/metrics HTTP/1.1
        \\host: local
        \\
        \\{"resourceMetrics":[{"resource":{"attributes":[{"key":"service.name","value":{"stringValue":"edge-runtime"}}]},"scopeMetrics":[{"scope":{"name":"demo"},"metrics":[{"name":"latency_ms","unit":"ms","gauge":{"dataPoints":[{"timeUnixNano":"124","asDouble":14.5}]}}]}]}]}
    ;
    const metrics_response = try responseForRequestWithState(
        std.testing.allocator,
        metrics_request,
        .{},
        &state,
    );
    defer std.testing.allocator.free(metrics_response);

    try std.testing.expect(std.mem.indexOf(u8, metrics_response, "HTTP/1.1 200 OK") != null);
    try std.testing.expectEqual(@as(usize, 1), state.telemetry_store.metrics.items.len);
    try std.testing.expectEqualStrings("edge-runtime", state.telemetry_store.metrics.items[0].source);
    try std.testing.expectEqualStrings("latency_ms", state.telemetry_store.metrics.items[0].name);

    const otlp_logs = try responseForRequestWithState(
        std.testing.allocator,
        "GET /telemetry/otlp/logs HTTP/1.1\r\nhost: local\r\n\r\n",
        .{ .node_id = "edge-a" },
        &state,
    );
    defer std.testing.allocator.free(otlp_logs);

    try std.testing.expect(std.mem.indexOf(u8, otlp_logs, "\"resourceLogs\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, otlp_logs, "\"service.instance.id\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, otlp_logs, "\"frame dropped\"") != null);

    const otlp_metrics = try responseForRequestWithState(
        std.testing.allocator,
        "GET /telemetry/otlp/metrics HTTP/1.1\r\nhost: local\r\n\r\n",
        .{ .node_id = "edge-a" },
        &state,
    );
    defer std.testing.allocator.free(otlp_metrics);

    try std.testing.expect(std.mem.indexOf(u8, otlp_metrics, "\"resourceMetrics\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, otlp_metrics, "\"latency_ms\"") != null);

    const traces_request =
        \\POST /v1/traces HTTP/1.1
        \\host: local
        \\
        \\{"resourceSpans":[{"resource":{"attributes":[{"key":"service.name","value":{"stringValue":"edge-runtime"}}]},"scopeSpans":[{"scope":{"name":"demo"},"spans":[{"traceId":"5b8efff798038103d269b633813fc60c","spanId":"eee19b7ec3c1b174","name":"invoke","kind":2,"startTimeUnixNano":"125","endTimeUnixNano":"175","status":{"code":1}}]}]}]}
    ;
    const traces_response = try responseForRequestWithState(
        std.testing.allocator,
        traces_request,
        .{},
        &state,
    );
    defer std.testing.allocator.free(traces_response);

    try std.testing.expect(std.mem.indexOf(u8, traces_response, "HTTP/1.1 200 OK") != null);
    try std.testing.expectEqual(@as(usize, 1), state.telemetry_store.traces.items.len);
    try std.testing.expectEqualStrings("edge-runtime", state.telemetry_store.traces.items[0].source);
    try std.testing.expectEqualStrings("invoke", state.telemetry_store.traces.items[0].name);
    try std.testing.expectEqual(telemetry.SpanKind.server, state.telemetry_store.traces.items[0].kind);

    const otlp_traces = try responseForRequestWithState(
        std.testing.allocator,
        "GET /telemetry/otlp/traces HTTP/1.1\r\nhost: local\r\n\r\n",
        .{ .node_id = "edge-a" },
        &state,
    );
    defer std.testing.allocator.free(otlp_traces);

    try std.testing.expect(std.mem.indexOf(u8, otlp_traces, "\"resourceSpans\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, otlp_traces, "\"traceId\":\"5b8efff798038103d269b633813fc60c\"") != null);
}

test "agent workload invoke reports missing workload" {
    var state = State{};
    defer state.deinit(std.testing.allocator);

    const response = try responseForRequestWithState(
        std.testing.allocator,
        "POST /workloads/42/invoke HTTP/1.1\r\nhost: local\r\n\r\n",
        .{},
        &state,
    );
    defer std.testing.allocator.free(response);

    try std.testing.expect(std.mem.indexOf(u8, response, "\"kind\":\"workload_invoke\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, response, "\"workload_id\":42") != null);
    try std.testing.expect(std.mem.indexOf(u8, response, "\"status\":\"missing\"") != null);
}

test "agent parses invoke args and stdin" {
    var parsed = try parseInvokeRequest(
        std.testing.allocator,
        \\{"args":[7, 11], "stdin":"payload"}
    ,
        .{},
    );
    defer parsed.deinit(std.testing.allocator);

    try std.testing.expectEqual(@as(usize, 2), parsed.args.len);
    try std.testing.expectEqual(@as(u32, 7), parsed.args[0]);
    try std.testing.expectEqual(@as(u32, 11), parsed.args[1]);
    try std.testing.expectEqualStrings("payload", parsed.stdin);

    var empty = try parseInvokeRequest(std.testing.allocator, "", .{});
    defer empty.deinit(std.testing.allocator);
    try std.testing.expectEqual(@as(usize, 0), empty.args.len);
    try std.testing.expectEqualStrings("", empty.stdin);
}

test "agent parses telemetry payloads" {
    const event = try parseTelemetryEventRequest(
        \\{"kind":"safety_stop","message":"manual stop","severity":"error"}
    ,
        .{},
    );
    try std.testing.expectEqualStrings("agent", event.source);
    try std.testing.expectEqualStrings("safety_stop", event.kind);
    try std.testing.expectEqual(telemetry.Severity.err, event.severity);

    const metric = try parseTelemetryMetricRequest(
        \\{"source":"gpu","name":"temperature_c","value":61.25,"unit":"celsius"}
    ,
        .{},
    );
    try std.testing.expectEqualStrings("gpu", metric.source);
    try std.testing.expectEqualStrings("temperature_c", metric.name);
    try std.testing.expectEqual(@as(f64, 61.25), metric.value);
    try std.testing.expectEqualStrings("celsius", metric.unit);

    const trace = try parseTelemetryTraceRequest(
        \\{"trace_id":"5b8efff798038103d269b633813fc60c","span_id":"eee19b7ec3c1b174","name":"invoke","kind":"client","start_time_ns":1,"end_time_ns":2,"status":"ok"}
    ,
        .{},
    );
    try std.testing.expectEqualStrings("agent", trace.source);
    try std.testing.expectEqualStrings("invoke", trace.name);
    try std.testing.expectEqual(telemetry.SpanKind.client, trace.kind);
    try std.testing.expectEqual(telemetry.SpanStatus.ok, trace.status);
}

test "agent runtime policy rejects oversized invoke inputs" {
    try std.testing.expectError(
        error.InvokeArgLimitExceeded,
        parseInvokeRequest(
            std.testing.allocator,
            \\{"args":[1,2]}
        ,
            .{ .max_invoke_args = 1 },
        ),
    );

    try std.testing.expectError(
        error.InvokeStdinTooLarge,
        parseInvokeRequest(
            std.testing.allocator,
            \\{"stdin":"abcd"}
        ,
            .{ .max_stdin_bytes = 3 },
        ),
    );

    try std.testing.expectError(
        error.InvokeRequestBodyTooLarge,
        parseInvokeRequest(std.testing.allocator, "{}", .{ .max_request_body_bytes = 1 }),
    );
}

test "agent parses workload invoke path" {
    try std.testing.expectEqual(@as(u64, 7), try parseInvokeWorkloadId("/workloads/7/invoke"));
    try std.testing.expectError(error.NotInvokePath, parseInvokeWorkloadId("/workloads"));
    try std.testing.expectError(error.InvalidWorkloadId, parseInvokeWorkloadId("/workloads/nope/invoke"));
}

test "agent parses workload check body path" {
    try std.testing.expectEqualStrings(
        "examples/workloads/tiny-mnist",
        try parseWorkloadCheckPath("{\"path\":\"examples/workloads/tiny-mnist\"}"),
    );
    try std.testing.expectError(error.MissingJsonField, parseWorkloadCheckPath("{\"workload\":\"x\"}"));
}

test "listen address parser accepts localhost and wildcard aliases" {
    const localhost = try parseListenAddress("localhost:7070");
    try std.testing.expectEqual(@as(u16, 7070), localhost.getPort());

    const wildcard = try parseListenAddress("*:8080");
    try std.testing.expectEqual(@as(u16, 8080), wildcard.getPort());
}
