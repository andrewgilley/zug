const std = @import("std");
const wit = @import("component/wit.zig");
const component_smoke = @import("component/component_smoke_wit.zig");
const edge_inference = @import("component/edge_inference_wit.zig");

pub const Options = struct {
    descriptor: []const u8,
    out_path: ?[]const u8 = null,
};

pub fn run(allocator: std.mem.Allocator, options: Options) !void {
    const rendered = try renderDescriptorAlloc(allocator, options.descriptor);
    defer allocator.free(rendered);

    if (options.out_path) |out_path| {
        try std.Io.Dir.cwd().writeFile(std.Options.debug_io, .{
            .sub_path = out_path,
            .data = rendered,
        });
        return;
    }

    std.debug.print("{s}", .{rendered});
}

pub fn renderDescriptorAlloc(allocator: std.mem.Allocator, descriptor: []const u8) ![]u8 {
    const base_name = std.fs.path.basename(descriptor);

    if (std.mem.eql(u8, descriptor, "edge-inference") or
        std.mem.eql(u8, descriptor, "edge_inference") or
        std.mem.eql(u8, base_name, "edge_inference_wit.zig"))
    {
        return wit.renderAlloc(allocator, edge_inference.package);
    }

    if (std.mem.eql(u8, descriptor, "component-smoke") or
        std.mem.eql(u8, descriptor, "component_smoke") or
        std.mem.eql(u8, base_name, "component_smoke_wit.zig"))
    {
        return wit.renderAlloc(allocator, component_smoke.package);
    }

    return error.UnknownWitDescriptor;
}

test "renders edge inference descriptor by stable name" {
    const allocator = std.testing.allocator;
    const rendered = try renderDescriptorAlloc(allocator, "edge-inference");
    defer allocator.free(rendered);

    try std.testing.expect(std.mem.indexOf(u8, rendered, "interface inference {") != null);
    try std.testing.expect(std.mem.indexOf(u8, rendered, "export run: func() -> result<_, inference.inference-error>;") != null);
}

test "renders edge inference descriptor by descriptor path basename" {
    const allocator = std.testing.allocator;
    const rendered = try renderDescriptorAlloc(allocator, "src/component/edge_inference_wit.zig");
    defer allocator.free(rendered);

    try std.testing.expect(std.mem.startsWith(u8, rendered, "package zug:edge-inference@0.1.0;"));
}

test "renders component smoke descriptor by stable name" {
    const allocator = std.testing.allocator;
    const rendered = try renderDescriptorAlloc(allocator, "component-smoke");
    defer allocator.free(rendered);

    try std.testing.expect(std.mem.startsWith(u8, rendered, "package zug:component-smoke@0.1.0;"));
    try std.testing.expect(std.mem.indexOf(u8, rendered, "world component-smoke") != null);
    try std.testing.expect(std.mem.indexOf(u8, rendered, "export run: func(") != null);
    try std.testing.expect(std.mem.indexOf(u8, rendered, "value: s32") != null);
    try std.testing.expect(std.mem.indexOf(u8, rendered, ") -> s32;") != null);
}

test "renders component smoke descriptor by descriptor path basename" {
    const allocator = std.testing.allocator;
    const rendered = try renderDescriptorAlloc(allocator, "src/component/component_smoke_wit.zig");
    defer allocator.free(rendered);

    try std.testing.expect(std.mem.startsWith(u8, rendered, "package zug:component-smoke@0.1.0;"));
}

test "checked-in edge inference WIT is generated from descriptor" {
    const allocator = std.testing.allocator;
    const rendered = try renderDescriptorAlloc(allocator, "edge-inference");
    defer allocator.free(rendered);

    const checked_in = try std.Io.Dir.cwd().readFileAlloc(
        std.Options.debug_io,
        "wit/edge-inference.wit",
        allocator,
        .limited(64 * 1024),
    );
    defer allocator.free(checked_in);

    try std.testing.expectEqualStrings(checked_in, rendered);
}
