const std = @import("std");
const network = @import("network.zig");
const target_profile = @import("target.zig");

const net = std.Io.net;

pub const Config = struct {
    node_id: []const u8 = "local-node",
    profile_name: []const u8 = "vision-f32-basic",
    listen: []const u8 = "127.0.0.1:7070",
    once: bool = false,
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

    std.debug.print("zug agent listening on {f}\n", .{server.socket.address});

    while (true) {
        const stream = try server.accept(io);
        try handleConnection(allocator, io, stream, config);
        if (config.once) break;
    }
}

fn handleConnection(
    allocator: std.mem.Allocator,
    io: std.Io,
    stream: net.Stream,
    config: Config,
) !void {
    defer stream.close(io);

    var read_buffer: [4096]u8 = undefined;
    var reader_state = stream.reader(io, &read_buffer);
    var request_buffer: [8192]u8 = undefined;
    const request = try readHttpRequest(&reader_state.interface, &request_buffer);

    const response = try responseForRequest(allocator, request, config);
    defer allocator.free(response);

    var write_buffer: [4096]u8 = undefined;
    var writer_state = stream.writer(io, &write_buffer);
    try writer_state.interface.writeAll(response);
    try writer_state.interface.flush();
}

fn readHttpRequest(reader: *std.Io.Reader, buffer: []u8) ![]const u8 {
    var len: usize = 0;

    while (len < buffer.len) {
        const read_count = try reader.readSliceShort(buffer[len .. len + 1]);
        if (read_count == 0) break;
        len += read_count;

        const request = buffer[0..len];
        if (std.mem.indexOf(u8, request, "\r\n\r\n") != null or
            std.mem.indexOf(u8, request, "\n\n") != null)
        {
            return request;
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
    const line = parseRequestLine(request) catch {
        return jsonResponse(allocator, .bad_request, "{\"error\":\"bad_request\"}");
    };

    if (!std.mem.eql(u8, line.method, "GET")) {
        return jsonResponse(allocator, .method_not_allowed, "{\"error\":\"method_not_allowed\"}");
    }

    const path = requestPath(line.target);
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

    return jsonResponse(allocator, .not_found, "{\"error\":\"not_found\"}");
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

    try json.endObject();
    return try out.toOwnedSlice();
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
    const body = try capabilitiesJson(std.testing.allocator, .{
        .node_id = "edge-a",
        .profile_name = "vision-f32-basic",
    });
    defer std.testing.allocator.free(body);

    try std.testing.expect(std.mem.indexOf(u8, body, "\"node_id\":\"edge-a\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, body, "\"profile\":\"vision-f32-basic\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, body, "\"http\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, body, "\"tcp\"") != null);
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

    const capabilities = try responseForRequest(
        std.testing.allocator,
        "GET /capabilities HTTP/1.1\r\nhost: local\r\n\r\n",
        .{ .node_id = "edge-a" },
    );
    defer std.testing.allocator.free(capabilities);
    try std.testing.expect(std.mem.indexOf(u8, capabilities, "\"agent_protocol\":\"zug-agent/v0\"") != null);

    const missing = try responseForRequest(
        std.testing.allocator,
        "GET /missing HTTP/1.1\r\nhost: local\r\n\r\n",
        .{},
    );
    defer std.testing.allocator.free(missing);
    try std.testing.expect(std.mem.indexOf(u8, missing, "HTTP/1.1 404 Not Found") != null);
}

test "listen address parser accepts localhost and wildcard aliases" {
    const localhost = try parseListenAddress("localhost:7070");
    try std.testing.expectEqual(@as(u16, 7070), localhost.getPort());

    const wildcard = try parseListenAddress("*:8080");
    try std.testing.expectEqual(@as(u16, 8080), wildcard.getPort());
}
