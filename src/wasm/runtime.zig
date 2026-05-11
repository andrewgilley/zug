const std = @import("std");

pub const binary = @import("binary.zig");
pub const component = @import("component.zig");
pub const fixtures = @import("fixtures.zig");
pub const imports = @import("imports.zig");
pub const instance = @import("instance.zig");
pub const interpreter = @import("interpreter.zig");
pub const module = @import("module.zig");
pub const validator = @import("validator.zig");

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

    pub fn parseComponent(self: Runtime, bytes: []const u8) !component.Component {
        return component.Component.parse(self.allocator, bytes);
    }

    pub fn instantiate(
        self: Runtime,
        parsed_module: *const module.Module,
        initial_memory_bytes: usize,
    ) !instance.Instance {
        try self.validateModule(parsed_module);

        return instance.Instance.init(self.allocator, parsed_module, initial_memory_bytes);
    }

    pub fn validateModule(self: Runtime, parsed_module: *const module.Module) !void {
        try validator.validate(self.allocator, parsed_module);
    }

    pub fn instantiateStarted(
        self: Runtime,
        parsed_module: *const module.Module,
        initial_memory_bytes: usize,
    ) !instance.Instance {
        var wasm_instance = try self.instantiate(parsed_module, initial_memory_bytes);
        errdefer wasm_instance.deinit();

        var wasm_interpreter = interpreter.Interpreter.init(&wasm_instance);
        try wasm_interpreter.runStart();

        return wasm_instance;
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

test "runtime instantiates and runs module start function" {
    const allocator = std.testing.allocator;

    const runtime = Runtime.init(allocator);

    var parsed = try runtime.parseModule(fixtures.globals_and_start);
    defer parsed.deinit(allocator);

    var wasm_instance = try runtime.instantiateStarted(&parsed, 64 * 1024);
    defer wasm_instance.deinit();

    var wasm_interpreter = interpreter.Interpreter.init(&wasm_instance);
    const result = (try wasm_interpreter.callExport("run", &.{})) orelse return error.MissingReturnValue;

    try std.testing.expectEqual(@as(u32, 12), switch (result) {
        .i32 => |value| value,
        else => return error.ExpectedI32Value,
    });
}
