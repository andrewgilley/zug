const std = @import("std");
const imports = @import("imports.zig");
const module = @import("module.zig");
const wasi_nn_abi = @import("wasi_nn_abi");

pub const Instance = struct {
    allocator: std.mem.Allocator,
    module: *const module.Module,
    memory_bytes: []u8,
    memory: wasi_nn_abi.LinearMemory,
    import_resolver: ?*imports.Resolver = null,

    pub fn init(
        allocator: std.mem.Allocator,
        parsed_module: *const module.Module,
        initial_memory_bytes: usize,
    ) !Instance {
        if (initial_memory_bytes == 0) return error.InvalidMemorySize;

        const memory_bytes = try allocator.alloc(u8, initial_memory_bytes);
        errdefer allocator.free(memory_bytes);
        @memset(memory_bytes, 0);

        return .{
            .allocator = allocator,
            .module = parsed_module,
            .memory_bytes = memory_bytes,
            .memory = wasi_nn_abi.LinearMemory.init(memory_bytes),
        };
    }

    pub fn deinit(self: *Instance) void {
        self.allocator.free(self.memory_bytes);
        self.* = undefined;
    }

    pub fn bindImports(self: *Instance, resolver: *imports.Resolver) void {
        self.import_resolver = resolver;
    }
};

test "instance owns linear memory" {
    const allocator = std.testing.allocator;

    var parsed = try module.Module.parse(allocator, "\x00asm\x01\x00\x00\x00");
    defer parsed.deinit(allocator);

    var instance = try Instance.init(allocator, &parsed, 64 * 1024);
    defer instance.deinit();

    try instance.memory.writeU32(0, 42);
    try std.testing.expectEqual(@as(u32, 42), try instance.memory.readU32(0));
}
