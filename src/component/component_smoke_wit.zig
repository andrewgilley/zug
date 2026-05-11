const wit = @import("wit.zig");

const s32_type = wit.Type.primitiveType(.s32);

pub const package = wit.Package{
    .id = "zug:component-smoke@0.1.0",
    .worlds = &.{component_smoke_world},
};

pub const component_smoke_world = wit.World{
    .name = "component-smoke",
    .exports = &.{.{
        .function = .{
            .name = "run",
            .params = &.{.{ .name = "value", .ty = s32_type }},
            .result = s32_type,
        },
    }},
};

test "component smoke descriptor renders WIT" {
    const std = @import("std");
    const allocator = std.testing.allocator;
    const rendered = try wit.renderAlloc(allocator, package);
    defer allocator.free(rendered);

    try std.testing.expect(std.mem.startsWith(u8, rendered, "package zug:component-smoke@0.1.0;\n"));
    try std.testing.expect(std.mem.indexOf(u8, rendered, "world component-smoke") != null);
    try std.testing.expect(std.mem.indexOf(u8, rendered, "export run: func(") != null);
    try std.testing.expect(std.mem.indexOf(u8, rendered, "value: s32") != null);
    try std.testing.expect(std.mem.indexOf(u8, rendered, ") -> s32;") != null);
}
