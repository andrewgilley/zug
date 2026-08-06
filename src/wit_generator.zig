const std = @import("std");
const wit = @import("component/wit.zig");
const component_smoke = @import("component/component_smoke_wit.zig");

pub const Options = struct {
    descriptor: []const u8,
    out_path: ?[]const u8 = null,
};

pub fn run(allocator: std.mem.Allocator, options: Options) !void {
    const rendered = try renderDescriptorAlloc(allocator, options.descriptor);
    defer allocator.free(rendered);

    if (options.out_path) |out_path| {
        try std.Io.Dir.cwd().writeFile(std.Options.debug_io, .{ .sub_path = out_path, .data = rendered });
    } else {
        std.debug.print("{s}", .{rendered});
    }
}

pub fn renderDescriptorAlloc(allocator: std.mem.Allocator, descriptor: []const u8) ![]u8 {
    const base_name = std.fs.path.basename(descriptor);
    if (std.mem.eql(u8, descriptor, "component-smoke") or
        std.mem.eql(u8, descriptor, "component_smoke") or
        std.mem.eql(u8, base_name, "component_smoke_wit.zig"))
    {
        return wit.renderAlloc(allocator, component_smoke.package);
    }
    return error.UnknownWitDescriptor;
}

test "renders component smoke descriptor" {
    const allocator = std.testing.allocator;
    const rendered = try renderDescriptorAlloc(allocator, "component-smoke");
    defer allocator.free(rendered);
    try std.testing.expect(std.mem.startsWith(u8, rendered, "package zug:component-smoke@0.1.0;"));
    try std.testing.expect(std.mem.indexOf(u8, rendered, "world component-smoke") != null);
}
