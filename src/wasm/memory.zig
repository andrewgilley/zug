const std = @import("std");

pub const LinearMemory = struct {
    bytes: []u8,

    pub fn init(bytes: []u8) LinearMemory {
        return .{ .bytes = bytes };
    }

    pub fn read(self: *const LinearMemory, ptr: u32, len: u32) ![]const u8 {
        const range = try checkedRange(self.bytes.len, ptr, len);
        return self.bytes[range.start..range.end];
    }

    pub fn write(self: *LinearMemory, ptr: u32, data: []const u8) !void {
        const len = std.math.cast(u32, data.len) orelse return error.InvalidMemoryRange;
        const range = try checkedRange(self.bytes.len, ptr, len);
        @memcpy(self.bytes[range.start..range.end], data);
    }

    pub fn readU32(self: *const LinearMemory, ptr: u32) !u32 {
        const bytes = try self.read(ptr, 4);
        return @as(u32, bytes[0]) |
            (@as(u32, bytes[1]) << 8) |
            (@as(u32, bytes[2]) << 16) |
            (@as(u32, bytes[3]) << 24);
    }

    pub fn writeU32(self: *LinearMemory, ptr: u32, value: u32) !void {
        const bytes = try self.writeSlice(ptr, 4);
        bytes[0] = @truncate(value);
        bytes[1] = @truncate(value >> 8);
        bytes[2] = @truncate(value >> 16);
        bytes[3] = @truncate(value >> 24);
    }

    pub fn writeU64(self: *LinearMemory, ptr: u32, value: u64) !void {
        const bytes = try self.writeSlice(ptr, 8);
        for (bytes, 0..) |*byte, shift| byte.* = @truncate(value >> @intCast(shift * 8));
    }

    pub fn writeSlice(self: *LinearMemory, ptr: u32, len: u32) ![]u8 {
        const range = try checkedRange(self.bytes.len, ptr, len);
        return self.bytes[range.start..range.end];
    }
};

const Range = struct { start: usize, end: usize };

fn checkedRange(container_len: usize, ptr: u32, len: u32) !Range {
    const start = std.math.cast(usize, ptr) orelse return error.InvalidMemoryRange;
    const size = std.math.cast(usize, len) orelse return error.InvalidMemoryRange;
    const end = std.math.add(usize, start, size) catch return error.InvalidMemoryRange;
    if (end > container_len) return error.InvalidMemoryRange;
    return .{ .start = start, .end = end };
}
