const std = @import("std");

pub const FileUploadInput = struct {
    path: []const u8,
    offset: u64,
    data_hex: []const u8,
    truncate: bool = false,
};

pub const WriteResult = struct {
    stored_path: []const u8,
    byte_len: usize,

    pub fn deinit(self: *WriteResult, allocator: std.mem.Allocator) void {
        allocator.free(self.stored_path);
        self.* = undefined;
    }
};

pub fn validateRelativePath(path: []const u8) !void {
    if (path.len == 0) return error.MissingUploadPath;
    if (std.fs.path.isAbsolute(path)) return error.InvalidUploadPath;
    if (std.mem.indexOfScalar(u8, path, '\\') != null) return error.InvalidUploadPath;
    if (std.mem.indexOfScalar(u8, path, ':') != null) return error.InvalidUploadPath;

    var parts = std.mem.splitScalar(u8, path, '/');
    while (parts.next()) |part| {
        if (part.len == 0) return error.InvalidUploadPath;
        if (std.mem.eql(u8, part, ".") or std.mem.eql(u8, part, "..")) {
            return error.InvalidUploadPath;
        }
    }
}

pub fn storedPath(
    allocator: std.mem.Allocator,
    artifact_root: []const u8,
    relative_path: []const u8,
) ![]u8 {
    try validateRelativePath(relative_path);
    return try std.fs.path.join(allocator, &.{ artifact_root, relative_path });
}

pub fn decodeHexAlloc(
    allocator: std.mem.Allocator,
    value: []const u8,
    max_decoded_bytes: usize,
) ![]u8 {
    if (value.len % 2 != 0) return error.InvalidUploadHex;
    const decoded_len = value.len / 2;
    if (decoded_len > max_decoded_bytes) return error.UploadChunkTooLarge;

    const out = try allocator.alloc(u8, decoded_len);
    errdefer allocator.free(out);

    for (out, 0..) |*byte, index| {
        const hi = try hexNibble(value[index * 2]);
        const lo = try hexNibble(value[index * 2 + 1]);
        byte.* = (hi << 4) | lo;
    }

    return out;
}

pub fn writeChunk(
    allocator: std.mem.Allocator,
    artifact_root: []const u8,
    input: FileUploadInput,
    max_decoded_bytes: usize,
) !WriteResult {
    const path = try storedPath(allocator, artifact_root, input.path);
    errdefer allocator.free(path);

    const decoded = try decodeHexAlloc(allocator, input.data_hex, max_decoded_bytes);
    defer allocator.free(decoded);

    const io = std.Options.debug_io;
    if (std.fs.path.dirname(path)) |parent| {
        try std.Io.Dir.cwd().createDirPath(io, parent);
    }

    var file = try std.Io.Dir.cwd().createFile(io, path, .{
        .truncate = input.truncate,
        .read = true,
    });
    defer file.close(io);

    try file.writePositionalAll(io, decoded, input.offset);

    return .{
        .stored_path = path,
        .byte_len = decoded.len,
    };
}

fn hexNibble(byte: u8) !u8 {
    return switch (byte) {
        '0'...'9' => byte - '0',
        'a'...'f' => byte - 'a' + 10,
        'A'...'F' => byte - 'A' + 10,
        else => error.InvalidUploadHex,
    };
}

test "package transfer rejects unsafe relative paths" {
    try validateRelativePath("workloads/demo/model.onnx");
    try std.testing.expectError(error.InvalidUploadPath, validateRelativePath("../model.onnx"));
    try std.testing.expectError(error.InvalidUploadPath, validateRelativePath("workloads//model.onnx"));
    try std.testing.expectError(error.InvalidUploadPath, validateRelativePath("C:/model.onnx"));
    try std.testing.expectError(error.InvalidUploadPath, validateRelativePath("workloads\\model.onnx"));
}

test "package transfer decodes bounded hex chunks" {
    const bytes = try decodeHexAlloc(std.testing.allocator, "68656c6c6f", 8);
    defer std.testing.allocator.free(bytes);

    try std.testing.expectEqualStrings("hello", bytes);
    try std.testing.expectError(error.InvalidUploadHex, decodeHexAlloc(std.testing.allocator, "abc", 8));
    try std.testing.expectError(error.UploadChunkTooLarge, decodeHexAlloc(std.testing.allocator, "000102", 2));
}
