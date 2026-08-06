const std = @import("std");
const imports = @import("imports.zig");
const module = @import("module.zig");

pub const max_manifest_bytes = 64 * 1024;

pub const RuntimeCapabilities = struct {
    max_memory_bytes: usize = 256 * 1024 * 1024,

    pub fn supportsImport(self: RuntimeCapabilities, module_name: []const u8, function_name: []const u8) bool {
        _ = self;
        return imports.Resolver.resolve(module_name, function_name) != null;
    }
};

pub const RuntimeRequest = struct {
    export_name: []const u8,
    initial_memory_bytes: usize,
};

pub const RequiredImport = struct {
    module_name: []const u8,
    function_name: []const u8,
};

pub const Manifest = struct {
    name: ?[]const u8 = null,
    runtime: ?[]const u8 = null,
    export_name: ?[]const u8 = null,
    min_memory_bytes: ?usize = null,
    max_memory_bytes: ?usize = null,
    required_imports: std.ArrayList(RequiredImport) = .empty,

    pub fn parse(allocator: std.mem.Allocator, bytes: []const u8) !Manifest {
        var manifest = Manifest{};
        errdefer manifest.deinit(allocator);

        var lines = std.mem.splitScalar(u8, bytes, '\n');
        while (lines.next()) |raw_line| {
            const without_comment = if (std.mem.indexOfScalar(u8, raw_line, '#')) |index|
                raw_line[0..index]
            else
                raw_line;
            const line = trim(without_comment);
            if (line.len == 0) continue;

            const equals_index = std.mem.indexOfScalar(u8, line, '=') orelse return error.InvalidManifestLine;
            const key = trim(line[0..equals_index]);
            const value = trim(line[equals_index + 1 ..]);
            if (key.len == 0 or value.len == 0) return error.InvalidManifestLine;

            if (std.mem.eql(u8, key, "name")) {
                try replaceString(allocator, &manifest.name, value);
            } else if (std.mem.eql(u8, key, "runtime")) {
                try replaceString(allocator, &manifest.runtime, value);
            } else if (std.mem.eql(u8, key, "export")) {
                try replaceString(allocator, &manifest.export_name, value);
            } else if (std.mem.eql(u8, key, "min_memory_bytes")) {
                manifest.min_memory_bytes = try parseByteCount(value);
            } else if (std.mem.eql(u8, key, "max_memory_bytes")) {
                manifest.max_memory_bytes = try parseByteCount(value);
            } else if (std.mem.eql(u8, key, "requires_import")) {
                try manifest.appendRequiredImport(allocator, value);
            } else {
                return error.UnknownManifestKey;
            }
        }

        return manifest;
    }

    pub fn deinit(self: *Manifest, allocator: std.mem.Allocator) void {
        freeOptional(allocator, self.name);
        freeOptional(allocator, self.runtime);
        freeOptional(allocator, self.export_name);
        for (self.required_imports.items) |required| {
            allocator.free(required.module_name);
            allocator.free(required.function_name);
        }
        self.required_imports.deinit(allocator);
        self.* = undefined;
    }

    pub fn validate(
        self: *const Manifest,
        parsed_module: *const module.Module,
        caps: RuntimeCapabilities,
        request: RuntimeRequest,
    ) !void {
        if (!hasFunctionExport(parsed_module, request.export_name)) return error.ManifestExportMissing;
        if (request.initial_memory_bytes > caps.max_memory_bytes) return error.ManifestMemoryTooLarge;
        if (self.min_memory_bytes) |minimum| {
            if (request.initial_memory_bytes < minimum) return error.ManifestMemoryTooSmall;
        }
        if (self.max_memory_bytes) |maximum| {
            if (request.initial_memory_bytes > maximum) return error.ManifestMemoryTooLarge;
        }

        for (self.required_imports.items) |required| {
            if (!caps.supportsImport(required.module_name, required.function_name)) {
                return error.ManifestUnsupportedImport;
            }
            if (!hasFunctionImport(parsed_module, required.module_name, required.function_name)) {
                return error.ManifestImportMissingFromModule;
            }
        }
    }

    fn appendRequiredImport(self: *Manifest, allocator: std.mem.Allocator, spec: []const u8) !void {
        const separator = std.mem.indexOfScalar(u8, spec, '.') orelse return error.InvalidManifestImport;
        if (separator == 0 or separator + 1 >= spec.len) return error.InvalidManifestImport;

        const module_name = try allocator.dupe(u8, trim(spec[0..separator]));
        errdefer allocator.free(module_name);
        const function_name = try allocator.dupe(u8, trim(spec[separator + 1 ..]));
        errdefer allocator.free(function_name);
        try self.required_imports.append(allocator, .{
            .module_name = module_name,
            .function_name = function_name,
        });
    }
};

fn hasFunctionExport(parsed_module: *const module.Module, export_name: []const u8) bool {
    for (parsed_module.exports.items) |exported| {
        if (exported.kind == .function and std.mem.eql(u8, exported.name, export_name)) return true;
    }
    return false;
}

fn hasFunctionImport(parsed_module: *const module.Module, module_name: []const u8, function_name: []const u8) bool {
    for (parsed_module.imports.items) |import| {
        if (import.kind == .function and
            std.mem.eql(u8, import.module, module_name) and
            std.mem.eql(u8, import.name, function_name)) return true;
    }
    return false;
}

fn replaceString(allocator: std.mem.Allocator, target: *?[]const u8, value: []const u8) !void {
    if (target.*) |old| allocator.free(old);
    target.* = try allocator.dupe(u8, value);
}

fn freeOptional(allocator: std.mem.Allocator, value: ?[]const u8) void {
    if (value) |actual| allocator.free(actual);
}

fn parseByteCount(value: []const u8) !usize {
    const suffixes = [_]struct { suffix: []const u8, multiplier: usize }{
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
    return std.mem.trim(u8, value, " \t\r\"");
}

test "manifest parses generic runtime requirements" {
    const allocator = std.testing.allocator;
    var manifest = try Manifest.parse(allocator,
        \\name = example
        \\runtime = zug-0.1
        \\export = run
        \\min_memory_bytes = 64KiB
        \\max_memory_bytes = 16MiB
        \\requires_import = wasi_snapshot_preview1.fd_write
        \\
    );
    defer manifest.deinit(allocator);

    try std.testing.expectEqualStrings("example", manifest.name.?);
    try std.testing.expectEqual(@as(usize, 64 * 1024), manifest.min_memory_bytes.?);
    try std.testing.expectEqual(@as(usize, 1), manifest.required_imports.items.len);
}
