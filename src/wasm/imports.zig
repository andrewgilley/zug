const std = @import("std");
const wasi_nn_abi = @import("wasi_nn_abi");

pub const wasi_nn_module_name = "wasi_nn";
pub const wasi_module_name = "wasi_snapshot_preview1";

pub const Function = enum {
    load_graph,
    init_execution_context,
    set_input_by_index,
    compute,
    get_output_descriptor,
    get_output,
    fd_write,
    proc_exit,
};

pub const Resolver = struct {
    surface: ?*wasi_nn_abi.Surface = null,
    allocator: ?std.mem.Allocator = null,
    memory: ?*wasi_nn_abi.LinearMemory = null,
    stdout: std.ArrayList(u8) = .empty,
    stderr: std.ArrayList(u8) = .empty,
    exit_code: ?u32 = null,

    pub fn init(surface: *wasi_nn_abi.Surface) Resolver {
        return .{
            .surface = surface,
            .allocator = surface.allocator,
            .memory = surface.memory,
        };
    }

    pub fn initWasi(allocator: std.mem.Allocator, memory: *wasi_nn_abi.LinearMemory) Resolver {
        return .{
            .allocator = allocator,
            .memory = memory,
        };
    }

    pub fn deinit(self: *Resolver) void {
        if (self.allocator) |allocator| {
            self.stdout.deinit(allocator);
            self.stderr.deinit(allocator);
        }
        self.* = undefined;
    }

    pub fn resolve(module_name: []const u8, function_name: []const u8) ?Function {
        if (std.mem.eql(u8, module_name, wasi_nn_module_name)) {
            if (std.mem.eql(u8, function_name, "load_graph")) return .load_graph;
            if (std.mem.eql(u8, function_name, "init_execution_context")) return .init_execution_context;
            if (std.mem.eql(u8, function_name, "set_input_by_index")) return .set_input_by_index;
            if (std.mem.eql(u8, function_name, "compute")) return .compute;
            if (std.mem.eql(u8, function_name, "get_output_descriptor")) return .get_output_descriptor;
            if (std.mem.eql(u8, function_name, "get_output")) return .get_output;

            return null;
        }

        if (std.mem.eql(u8, module_name, wasi_module_name)) {
            if (std.mem.eql(u8, function_name, "fd_write")) return .fd_write;
            if (std.mem.eql(u8, function_name, "proc_exit")) return .proc_exit;

            return null;
        }

        return null;
    }

    pub fn call(self: *Resolver, function: Function, args: []const u32) u32 {
        const status = switch (function) {
            .load_graph => blk: {
                if (args.len != 5) break :blk wasi_nn_abi.Status.runtime_error;
                const surface = self.surface orelse break :blk wasi_nn_abi.Status.runtime_error;
                break :blk surface.loadGraph(args[0], args[1], args[2], args[3], args[4]);
            },
            .init_execution_context => blk: {
                if (args.len != 2) break :blk wasi_nn_abi.Status.runtime_error;
                const surface = self.surface orelse break :blk wasi_nn_abi.Status.runtime_error;
                break :blk surface.initExecutionContext(args[0], args[1]);
            },
            .set_input_by_index => blk: {
                if (args.len != 7) break :blk wasi_nn_abi.Status.runtime_error;
                const surface = self.surface orelse break :blk wasi_nn_abi.Status.runtime_error;
                break :blk surface.setInputByIndex(
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
                const surface = self.surface orelse break :blk wasi_nn_abi.Status.runtime_error;
                break :blk surface.compute(args[0]);
            },
            .get_output_descriptor => blk: {
                if (args.len != 7) break :blk wasi_nn_abi.Status.runtime_error;
                const surface = self.surface orelse break :blk wasi_nn_abi.Status.runtime_error;
                break :blk surface.getOutputDescriptor(
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
                const surface = self.surface orelse break :blk wasi_nn_abi.Status.runtime_error;
                break :blk surface.getOutput(args[0], args[1], args[2], args[3], args[4]);
            },
            .fd_write => return self.fdWrite(args),
            .proc_exit => {
                if (args.len != 1) return wasiErrnoInval;
                self.exit_code = args[0];
                return 0;
            },
        };

        return @intFromEnum(status);
    }

    fn fdWrite(self: *Resolver, args: []const u32) u32 {
        if (args.len != 4) return wasiErrnoInval;

        const allocator = self.allocator orelse return wasiErrnoInval;
        const memory = self.memory orelse return wasiErrnoFault;
        const fd = args[0];
        const iovs_ptr = args[1];
        const iovs_len = args[2];
        const nwritten_ptr = args[3];

        var target: *std.ArrayList(u8) = switch (fd) {
            1 => &self.stdout,
            2 => &self.stderr,
            else => return wasiErrnoBadf,
        };

        var written: u32 = 0;
        for (0..iovs_len) |index| {
            const index_u32 = std.math.cast(u32, index) orelse return wasiErrnoInval;
            const iov_ptr = std.math.add(u32, iovs_ptr, index_u32 * 8) catch return wasiErrnoFault;
            const buf_ptr = memory.readU32(iov_ptr) catch return wasiErrnoFault;
            const buf_len = memory.readU32(iov_ptr + 4) catch return wasiErrnoFault;
            const bytes = memory.read(buf_ptr, buf_len) catch return wasiErrnoFault;

            target.appendSlice(allocator, bytes) catch return wasiErrnoIo;
            written = std.math.add(u32, written, buf_len) catch return wasiErrnoIo;
        }

        memory.writeU32(nwritten_ptr, written) catch return wasiErrnoFault;

        return wasiErrnoSuccess;
    }
};

const wasiErrnoSuccess = 0;
const wasiErrnoBadf = 8;
const wasiErrnoFault = 21;
const wasiErrnoInval = 28;
const wasiErrnoIo = 29;

test "resolver maps wasi-nn import names" {
    try std.testing.expectEqual(Function.load_graph, Resolver.resolve("wasi_nn", "load_graph").?);
    try std.testing.expectEqual(Function.compute, Resolver.resolve("wasi_nn", "compute").?);
    try std.testing.expect(Resolver.resolve("env", "compute") == null);
}

test "resolver maps wasi imports" {
    try std.testing.expectEqual(Function.fd_write, Resolver.resolve(wasi_module_name, "fd_write").?);
    try std.testing.expectEqual(Function.proc_exit, Resolver.resolve(wasi_module_name, "proc_exit").?);
}

test "resolver records proc_exit code" {
    var memory_bytes = [_]u8{0} ** 64;
    var memory = wasi_nn_abi.LinearMemory.init(&memory_bytes);
    var resolver = Resolver.initWasi(std.testing.allocator, &memory);
    defer resolver.deinit();

    try std.testing.expectEqual(@as(u32, 0), resolver.call(.proc_exit, &.{7}));
    try std.testing.expectEqual(@as(?u32, 7), resolver.exit_code);
}
