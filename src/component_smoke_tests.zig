const std = @import("std");
const wasm_interpreter = @import("wasm/interpreter.zig");
const wasm_runtime = @import("wasm/runtime.zig");

test "component smoke guest runs through core wasm runtime" {
    const allocator = std.testing.allocator;

    const bytes = try std.Io.Dir.cwd().readFileAlloc(
        std.Options.debug_io,
        "zig-out/component_smoke.wasm",
        allocator,
        .limited(1024 * 1024),
    );
    defer allocator.free(bytes);

    const runtime = wasm_runtime.Runtime.init(allocator);
    var parsed = try runtime.parseModule(bytes);
    defer parsed.deinit(allocator);

    var instance = try runtime.instantiate(&parsed, 64 * 1024);
    defer instance.deinit();

    var interpreter = wasm_interpreter.Interpreter.init(&instance);
    const result = (try interpreter.callExport("run", &.{.{ .i32 = 7 }})) orelse return error.MissingReturnValue;

    try std.testing.expectEqual(@as(u32, 42), switch (result) {
        .i32 => |value| value,
        else => return error.ExpectedI32Value,
    });
}
