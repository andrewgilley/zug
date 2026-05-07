const std = @import("std");
const network = @import("network.zig");
const target = @import("target.zig");

pub const max_manifest_bytes = 64 * 1024;
pub const manifest_file_name = "zug.toml";

pub const Requirements = struct {
    wasi_nn: bool = false,
    wasi_http: bool = false,
    memory_bytes: ?usize = null,
    dtype: ?target.DType = null,
    target_name: ?[]const u8 = null,
};

pub const Manifest = struct {
    name: ?[]const u8 = null,
    entrypoint: ?[]const u8 = null,
    wasm_path: ?[]const u8 = null,
    model_path: ?[]const u8 = null,
    model_encoding: ?[]const u8 = null,
    requires: Requirements = .{},
    network: network.Requirements = .{},

    pub fn parse(allocator: std.mem.Allocator, bytes: []const u8) !Manifest {
        var manifest = Manifest{};
        errdefer manifest.deinit(allocator);

        var section: Section = .root;
        var lines = std.mem.splitScalar(u8, bytes, '\n');
        while (lines.next()) |raw_line| {
            const line = trim(stripComment(raw_line));
            if (line.len == 0) continue;

            if (line[0] == '[') {
                if (line[line.len - 1] != ']') return error.InvalidWorkloadSection;
                section = try parseSection(trim(line[1 .. line.len - 1]));
                continue;
            }

            const equals_index = std.mem.indexOfScalar(u8, line, '=') orelse return error.InvalidWorkloadManifestLine;
            const key = trim(line[0..equals_index]);
            const value = try parseValue(line[equals_index + 1 ..]);
            if (key.len == 0 or value.len == 0) return error.InvalidWorkloadManifestLine;

            switch (section) {
                .root => try manifest.parseRootKey(allocator, key, value),
                .requires => try manifest.parseRequiresKey(allocator, key, value),
                .network => try manifest.parseNetworkKey(allocator, key, value),
            }
        }

        if (manifest.wasm_path == null) return error.MissingWorkloadWasm;
        return manifest;
    }

    pub fn deinit(self: *Manifest, allocator: std.mem.Allocator) void {
        freeOptional(allocator, self.name);
        freeOptional(allocator, self.entrypoint);
        freeOptional(allocator, self.wasm_path);
        freeOptional(allocator, self.model_path);
        freeOptional(allocator, self.model_encoding);
        freeOptional(allocator, self.requires.target_name);
        self.network.deinit(allocator);
        self.* = undefined;
    }

    pub fn entrypointName(self: Manifest) []const u8 {
        return self.entrypoint orelse "run";
    }

    pub fn targetName(self: Manifest) ?[]const u8 {
        return self.requires.target_name;
    }

    pub fn validateForTarget(self: Manifest, profile: target.Profile) !void {
        if (self.requires.wasi_nn and !profile.supports_wasi_nn) return error.WorkloadRequiresWasiNn;
        if (self.requires.wasi_http and !profile.supports_wasi_http) return error.WorkloadRequiresWasiHttp;
        if (self.requires.wasi_nn and self.model_path == null) return error.WorkloadRequiresModel;

        if (self.requires.memory_bytes) |memory_bytes| {
            if (memory_bytes > profile.max_memory_bytes) return error.WorkloadMemoryTooLarge;
        }

        if (self.requires.dtype) |dtype| {
            if (!profile.supportsDType(dtype)) return error.WorkloadUnsupportedDType;
        }
        try self.network.validate(profile.network);

        if (self.model_encoding) |encoding| {
            if (!std.ascii.eqlIgnoreCase(encoding, "onnx")) return error.WorkloadUnsupportedModelEncoding;
        }
    }

    fn parseRootKey(self: *Manifest, allocator: std.mem.Allocator, key: []const u8, value: []const u8) !void {
        if (std.mem.eql(u8, key, "name")) {
            try replaceString(allocator, &self.name, value);
        } else if (std.mem.eql(u8, key, "entrypoint")) {
            try replaceString(allocator, &self.entrypoint, value);
        } else if (std.mem.eql(u8, key, "wasm")) {
            try replaceString(allocator, &self.wasm_path, value);
        } else if (std.mem.eql(u8, key, "model")) {
            try replaceString(allocator, &self.model_path, value);
        } else if (std.mem.eql(u8, key, "model_encoding")) {
            try replaceString(allocator, &self.model_encoding, value);
        } else {
            return error.UnknownWorkloadManifestKey;
        }
    }

    fn parseRequiresKey(self: *Manifest, allocator: std.mem.Allocator, key: []const u8, value: []const u8) !void {
        if (std.mem.eql(u8, key, "wasi_nn")) {
            self.requires.wasi_nn = try parseBool(value);
        } else if (std.mem.eql(u8, key, "wasi_http")) {
            self.requires.wasi_http = try parseBool(value);
        } else if (std.mem.eql(u8, key, "memory")) {
            self.requires.memory_bytes = try parseByteCount(value);
        } else if (std.mem.eql(u8, key, "memory_bytes")) {
            self.requires.memory_bytes = try parseByteCount(value);
        } else if (std.mem.eql(u8, key, "memory_mb")) {
            self.requires.memory_bytes = try std.math.mul(usize, try std.fmt.parseInt(usize, value, 10), 1024 * 1024);
        } else if (std.mem.eql(u8, key, "dtype")) {
            self.requires.dtype = try target.parseDType(value);
        } else if (std.mem.eql(u8, key, "target")) {
            try replaceString(allocator, &self.requires.target_name, value);
        } else {
            return error.UnknownWorkloadManifestKey;
        }
    }

    fn parseNetworkKey(self: *Manifest, allocator: std.mem.Allocator, key: []const u8, value: []const u8) !void {
        if (std.mem.eql(u8, key, "input")) {
            self.network.input = try network.parseProtocol(value);
        } else if (std.mem.eql(u8, key, "output")) {
            self.network.output = try network.parseProtocol(value);
        } else if (std.mem.eql(u8, key, "ingress")) {
            try self.network.ingress.append(allocator, try network.parseProtocol(value));
        } else if (std.mem.eql(u8, key, "egress")) {
            try self.network.egress.append(allocator, try network.parseProtocol(value));
        } else if (std.mem.eql(u8, key, "max_inflight")) {
            self.network.max_inflight = try std.fmt.parseInt(usize, value, 10);
        } else if (std.mem.eql(u8, key, "request_timeout_ms")) {
            self.network.request_timeout_ms = try std.fmt.parseInt(u64, value, 10);
        } else if (std.mem.eql(u8, key, "public_egress")) {
            self.network.public_egress = try parseBool(value);
        } else {
            return error.UnknownWorkloadManifestKey;
        }
    }
};

pub const Loaded = struct {
    root_path: []const u8,
    manifest_path: []const u8,
    manifest: Manifest,
    wasm_path: []const u8,
    model_path: ?[]const u8 = null,

    pub fn load(allocator: std.mem.Allocator, root_path: []const u8) !Loaded {
        const manifest_path = try std.fs.path.join(allocator, &.{ root_path, manifest_file_name });
        errdefer allocator.free(manifest_path);

        const manifest_bytes = try std.Io.Dir.cwd().readFileAlloc(
            std.Options.debug_io,
            manifest_path,
            allocator,
            .limited(max_manifest_bytes),
        );
        defer allocator.free(manifest_bytes);

        var manifest = try Manifest.parse(allocator, manifest_bytes);
        errdefer manifest.deinit(allocator);

        const wasm_path = try resolvePath(allocator, root_path, manifest.wasm_path.?);
        errdefer allocator.free(wasm_path);

        const model_path = if (manifest.model_path) |model| try resolvePath(allocator, root_path, model) else null;
        errdefer if (model_path) |path| allocator.free(path);

        return .{
            .root_path = try allocator.dupe(u8, root_path),
            .manifest_path = manifest_path,
            .manifest = manifest,
            .wasm_path = wasm_path,
            .model_path = model_path,
        };
    }

    pub fn deinit(self: *Loaded, allocator: std.mem.Allocator) void {
        allocator.free(self.root_path);
        allocator.free(self.manifest_path);
        self.manifest.deinit(allocator);
        allocator.free(self.wasm_path);
        if (self.model_path) |path| allocator.free(path);
        self.* = undefined;
    }
};

const Section = enum {
    root,
    requires,
    network,
};

fn parseSection(value: []const u8) !Section {
    if (std.mem.eql(u8, value, "requires")) return .requires;
    if (std.mem.eql(u8, value, "network")) return .network;
    return error.UnknownWorkloadSection;
}

fn resolvePath(allocator: std.mem.Allocator, root_path: []const u8, path: []const u8) ![]const u8 {
    if (std.fs.path.isAbsolute(path)) return allocator.dupe(u8, path);
    return std.fs.path.join(allocator, &.{ root_path, path });
}

fn replaceString(allocator: std.mem.Allocator, field: *?[]const u8, value: []const u8) !void {
    if (field.*) |old| allocator.free(old);
    field.* = try allocator.dupe(u8, value);
}

fn freeOptional(allocator: std.mem.Allocator, value: ?[]const u8) void {
    if (value) |actual| allocator.free(actual);
}

fn parseValue(raw: []const u8) ![]const u8 {
    const value = trim(raw);
    if (value.len < 2) return value;

    const first = value[0];
    const last = value[value.len - 1];
    if ((first == '"' and last == '"') or (first == '\'' and last == '\'')) {
        return value[1 .. value.len - 1];
    }

    if (first == '"' or first == '\'' or last == '"' or last == '\'') return error.InvalidWorkloadString;
    return value;
}

fn stripComment(line: []const u8) []const u8 {
    var in_quote: ?u8 = null;
    for (line, 0..) |char, index| {
        if (in_quote) |quote| {
            if (char == quote) in_quote = null;
            continue;
        }

        if (char == '"' or char == '\'') {
            in_quote = char;
            continue;
        }

        if (char == '#') return line[0..index];
    }

    return line;
}

fn parseBool(value: []const u8) !bool {
    if (std.mem.eql(u8, value, "true") or std.mem.eql(u8, value, "yes") or std.mem.eql(u8, value, "1")) return true;
    if (std.mem.eql(u8, value, "false") or std.mem.eql(u8, value, "no") or std.mem.eql(u8, value, "0")) return false;
    return error.InvalidWorkloadBool;
}

fn parseByteCount(value: []const u8) !usize {
    const suffixes = [_]struct {
        suffix: []const u8,
        multiplier: usize,
    }{
        .{ .suffix = "KiB", .multiplier = 1024 },
        .{ .suffix = "MiB", .multiplier = 1024 * 1024 },
        .{ .suffix = "GiB", .multiplier = 1024 * 1024 * 1024 },
        .{ .suffix = "KB", .multiplier = 1000 },
        .{ .suffix = "MB", .multiplier = 1000 * 1000 },
        .{ .suffix = "GB", .multiplier = 1000 * 1000 * 1000 },
    };

    for (suffixes) |suffix| {
        if (std.mem.endsWith(u8, value, suffix.suffix)) {
            const number = try std.fmt.parseInt(usize, trim(value[0 .. value.len - suffix.suffix.len]), 10);
            return std.math.mul(usize, number, suffix.multiplier);
        }
    }

    return std.fmt.parseInt(usize, value, 10);
}

fn trim(value: []const u8) []const u8 {
    return std.mem.trim(u8, value, " \t\r");
}

test "workload manifest parses root requires and network sections" {
    const allocator = std.testing.allocator;

    var manifest = try Manifest.parse(allocator,
        \\name = "tiny-mnist"
        \\entrypoint = "run"
        \\wasm = "guest.wasm"
        \\model = "model.onnx"
        \\model_encoding = "onnx"
        \\
        \\[requires]
        \\wasi_nn = true
        \\memory_mb = 64
        \\dtype = "float32"
        \\target = "vision-f32-basic"
        \\
        \\[network]
        \\input = "local"
        \\output = "stdout"
        \\egress = "https"
        \\max_inflight = 4
        \\request_timeout_ms = 1000
        \\
    );
    defer manifest.deinit(allocator);

    try std.testing.expectEqualStrings("tiny-mnist", manifest.name.?);
    try std.testing.expectEqualStrings("guest.wasm", manifest.wasm_path.?);
    try std.testing.expect(manifest.requires.wasi_nn);
    try std.testing.expectEqual(@as(usize, 64 * 1024 * 1024), manifest.requires.memory_bytes.?);
    try std.testing.expectEqual(target.DType.float32, manifest.requires.dtype.?);
    try std.testing.expectEqual(network.Protocol.stdio, manifest.network.output.?);
    try std.testing.expectEqual(network.Protocol.https, manifest.network.egress.items[0]);
    try std.testing.expectEqual(@as(usize, 4), manifest.network.max_inflight.?);
}

test "workload manifest validates target requirements" {
    const allocator = std.testing.allocator;

    var manifest = try Manifest.parse(allocator,
        \\wasm = "guest.wasm"
        \\model = "model.onnx"
        \\[requires]
        \\wasi_nn = true
        \\dtype = "float32"
        \\target = "vision-f32-basic"
        \\
    );
    defer manifest.deinit(allocator);

    try manifest.validateForTarget(try target.resolve(manifest.targetName()));

    try std.testing.expectError(
        error.WorkloadRequiresWasiNn,
        manifest.validateForTarget(try target.resolve("wasm-basic"))
    );
}
