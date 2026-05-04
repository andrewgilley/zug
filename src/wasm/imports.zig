const std = @import("std");
const wasi_nn_abi = @import("wasi_nn_abi");

pub const wasi_nn_module_name = "wasi_nn";

pub const Function = enum {
    load_graph,
    init_execution_context,
    set_input_by_index,
    compute,
    get_output_descriptor,
    get_output,
};

pub const Resolver = struct {
    surface: *wasi_nn_abi.Surface,

    pub fn init(surface: *wasi_nn_abi.Surface) Resolver {
        return .{
            .surface = surface,
        };
    }

    pub fn resolve(module_name: []const u8, function_name: []const u8) ?Function {
        if (!std.mem.eql(u8, module_name, wasi_nn_module_name)) return null;

        if (std.mem.eql(u8, function_name, "load_graph")) return .load_graph;
        if (std.mem.eql(u8, function_name, "init_execution_context")) return .init_execution_context;
        if (std.mem.eql(u8, function_name, "set_input_by_index")) return .set_input_by_index;
        if (std.mem.eql(u8, function_name, "compute")) return .compute;
        if (std.mem.eql(u8, function_name, "get_output_descriptor")) return .get_output_descriptor;
        if (std.mem.eql(u8, function_name, "get_output")) return .get_output;

        return null;
    }

    pub fn call(self: *Resolver, function: Function, args: []const u32) u32 {
        const status = switch (function) {
            .load_graph => blk: {
                if (args.len != 5) break :blk wasi_nn_abi.Status.runtime_error;
                break :blk self.surface.loadGraph(args[0], args[1], args[2], args[3], args[4]);
            },
            .init_execution_context => blk: {
                if (args.len != 2) break :blk wasi_nn_abi.Status.runtime_error;
                break :blk self.surface.initExecutionContext(args[0], args[1]);
            },
            .set_input_by_index => blk: {
                if (args.len != 7) break :blk wasi_nn_abi.Status.runtime_error;
                break :blk self.surface.setInputByIndex(
                    args[0],
                    args[1],
                    args[2],
                    args[3],
                    args[4],
                    args[5],
                    args[6],
                );
            },
            .compute => blk: {
                if (args.len != 1) break :blk wasi_nn_abi.Status.runtime_error;
                break :blk self.surface.compute(args[0]);
            },
            .get_output_descriptor => blk: {
                if (args.len != 7) break :blk wasi_nn_abi.Status.runtime_error;
                break :blk self.surface.getOutputDescriptor(
                    args[0],
                    args[1],
                    args[2],
                    args[3],
                    args[4],
                    args[5],
                    args[6],
                );
            },
            .get_output => blk: {
                if (args.len != 5) break :blk wasi_nn_abi.Status.runtime_error;
                break :blk self.surface.getOutput(args[0], args[1], args[2], args[3], args[4]);
            },
        };

        return @intFromEnum(status);
    }
};

test "resolver maps wasi-nn import names" {
    try std.testing.expectEqual(Function.load_graph, Resolver.resolve("wasi_nn", "load_graph").?);
    try std.testing.expectEqual(Function.compute, Resolver.resolve("wasi_nn", "compute").?);
    try std.testing.expect(Resolver.resolve("env", "compute") == null);
}
