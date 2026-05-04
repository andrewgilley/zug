const std = @import("std");
const imports = @import("imports.zig");
const instance = @import("instance.zig");

pub const Value = union(enum) {
    i32: u32,
    i64: u64,
    f32: f32,
    f64: f64,
};

pub const Interpreter = struct {
    instance: *instance.Instance,

    pub fn init(wasm_instance: *instance.Instance) Interpreter {
        return .{
            .instance = wasm_instance,
        };
    }

    pub fn callExport(
        self: *Interpreter,
        name: []const u8,
        args: []const Value,
    ) !?Value {
        _ = self;
        _ = name;
        _ = args;

        return error.UnsupportedWasmExecution;
    }

    pub fn callImport(
        self: *Interpreter,
        function: imports.Function,
        args: []const u32,
    ) !u32 {
        const resolver = self.instance.import_resolver orelse return error.MissingImportResolver;
        return resolver.call(function, args);
    }
};

test "interpreter reports unsupported export execution" {
    const allocator = std.testing.allocator;
    const module = @import("module.zig");

    var parsed = try module.Module.parse(allocator, "\x00asm\x01\x00\x00\x00");
    defer parsed.deinit(allocator);

    var wasm_instance = try instance.Instance.init(allocator, &parsed, 64 * 1024);
    defer wasm_instance.deinit();

    var interpreter = Interpreter.init(&wasm_instance);
    try std.testing.expectError(
        error.UnsupportedWasmExecution,
        interpreter.callExport("run", &.{}),
    );
}
