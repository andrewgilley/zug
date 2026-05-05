const std = @import("std");
const network = @import("network.zig");

pub const DType = enum {
    float32,
    int64,
    int32,
    uint8,
    bool,
};

pub const Profile = struct {
    name: []const u8,
    default_memory_bytes: usize,
    max_memory_bytes: usize,
    max_model_bytes: usize,
    supports_wasi_nn: bool = false,
    supports_wasi_http: bool = false,
    supported_dtypes: []const DType = &.{},
    network: network.Capabilities = .{},

    pub fn supportsDType(self: Profile, dtype: DType) bool {
        for (self.supported_dtypes) |supported| {
            if (supported == dtype) return true;
        }

        return false;
    }
};

const no_dtypes = [_]DType{};
const f32_only = [_]DType{.float32};
const basic_model_dtypes = [_]DType{ .float32, .int64, .int32, .uint8, .bool };
const local_protocols = [_]network.Protocol{ .local, .stdio };
const edge_protocols = [_]network.Protocol{ .local, .stdio, .http, .https, .tcp, .nats, .mqtt };

pub fn resolve(name: ?[]const u8) !Profile {
    const actual = name orelse "wasi-nn-basic";

    if (std.mem.eql(u8, actual, "wasm-basic")) {
        return .{
            .name = "wasm-basic",
            .default_memory_bytes = 16 * 1024 * 1024,
            .max_memory_bytes = 64 * 1024 * 1024,
            .max_model_bytes = 0,
            .supported_dtypes = &no_dtypes,
            .network = .{
                .protocols = &local_protocols,
                .max_inflight = 1,
                .max_request_timeout_ms = 0,
            },
        };
    }

    if (std.mem.eql(u8, actual, "wasi-nn-basic")) {
        return .{
            .name = "wasi-nn-basic",
            .default_memory_bytes = 16 * 1024 * 1024,
            .max_memory_bytes = 128 * 1024 * 1024,
            .max_model_bytes = 100 * 1024 * 1024,
            .supports_wasi_nn = true,
            .supported_dtypes = &basic_model_dtypes,
            .network = .{
                .protocols = &local_protocols,
                .max_inflight = 1,
                .max_request_timeout_ms = 0,
            },
        };
    }

    if (std.mem.eql(u8, actual, "vision-f32-basic")) {
        return .{
            .name = "vision-f32-basic",
            .default_memory_bytes = 16 * 1024 * 1024,
            .max_memory_bytes = 128 * 1024 * 1024,
            .max_model_bytes = 100 * 1024 * 1024,
            .supports_wasi_nn = true,
            .supported_dtypes = &f32_only,
            .network = .{
                .protocols = &edge_protocols,
                .max_inflight = 16,
                .max_request_timeout_ms = 30_000,
            },
        };
    }

    return error.UnknownTargetProfile;
}

pub fn parseDType(value: []const u8) !DType {
    if (std.ascii.eqlIgnoreCase(value, "float32") or std.ascii.eqlIgnoreCase(value, "f32")) return .float32;
    if (std.ascii.eqlIgnoreCase(value, "int64") or std.ascii.eqlIgnoreCase(value, "i64")) return .int64;
    if (std.ascii.eqlIgnoreCase(value, "int32") or std.ascii.eqlIgnoreCase(value, "i32")) return .int32;
    if (std.ascii.eqlIgnoreCase(value, "uint8") or std.ascii.eqlIgnoreCase(value, "u8")) return .uint8;
    if (std.ascii.eqlIgnoreCase(value, "bool")) return .bool;

    return error.UnknownTargetDType;
}

test "target profiles expose runtime resource limits" {
    const profile = try resolve("vision-f32-basic");

    try std.testing.expect(profile.supports_wasi_nn);
    try std.testing.expect(profile.supportsDType(.float32));
    try std.testing.expect(!profile.supportsDType(.int64));
    try std.testing.expectEqual(@as(usize, 128 * 1024 * 1024), profile.max_memory_bytes);
    try std.testing.expect(profile.network.supportsProtocol(.http));
    try std.testing.expect(profile.network.supportsProtocol(.nats));
}

test "target dtype parser accepts common aliases" {
    try std.testing.expectEqual(DType.float32, try parseDType("f32"));
    try std.testing.expectEqual(DType.int64, try parseDType("INT64"));
    try std.testing.expectError(error.UnknownTargetDType, parseDType("float16"));
}
