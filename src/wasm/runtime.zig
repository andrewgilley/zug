const std = @import("std");

pub const binary = @import("binary.zig");
pub const imports = @import("imports.zig");
pub const instance = @import("instance.zig");
pub const interpreter = @import("interpreter.zig");
pub const module = @import("module.zig");

pub const Runtime = struct {
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) Runtime {
        return .{
            .allocator = allocator,
        };
    }

    pub fn parseModule(self: Runtime, bytes: []const u8) !module.Module {
        return module.Module.parse(self.allocator, bytes);
    }

    pub fn instantiate(
        self: Runtime,
        parsed_module: *const module.Module,
        initial_memory_bytes: usize,
    ) !instance.Instance {
        return instance.Instance.init(self.allocator, parsed_module, initial_memory_bytes);
    }
};

test "runtime parses and instantiates an empty module" {
    const allocator = std.testing.allocator;

    const runtime = Runtime.init(allocator);

    var parsed = try runtime.parseModule("\x00asm\x01\x00\x00\x00");
    defer parsed.deinit(allocator);

    var wasm_instance = try runtime.instantiate(&parsed, 64 * 1024);
    defer wasm_instance.deinit();

    try wasm_instance.memory.writeU32(8, 7);
    try std.testing.expectEqual(@as(u32, 7), try wasm_instance.memory.readU32(8));
}
