const std = @import("std");
const wasi_nn_abi = @import("../wasi_nn_abi.zig");

pub const wasi_nn_module_name = "wasi_nn";
pub const zug_nn_module_name = "zug_nn";
pub const wasi_module_name = "wasi_snapshot_preview1";

pub const Function = enum {
    load_graph,
    load_preloaded_graph,
    init_execution_context,
    set_input_by_index,
    compute,
    get_output_descriptor,
    get_output,
    args_sizes_get,
    args_get,
    environ_sizes_get,
    environ_get,
    clock_time_get,
    random_get,
    fd_fdstat_get,
    fd_write,
    proc_exit,
};

pub const Arg = union(enum) {
    i32: u32,
    i64: u64,
    f32: f32,
    f64: f64,
};

pub const WasiConfig = struct {
    args: []const []const u8 = &.{},
    environ: []const []const u8 = &.{},
};

pub const Resolver = struct {
    surface: ?*wasi_nn_abi.Surface = null,
    allocator: ?std.mem.Allocator = null,
    memory: ?*wasi_nn_abi.LinearMemory = null,
    args: []const []const u8 = &.{},
    environ: []const []const u8 = &.{},
    stdout: std.ArrayList(u8) = .empty,
    stderr: std.ArrayList(u8) = .empty,
    exit_code: ?u32 = null,
    fallback_clock_ns: u64 = 1,

    pub fn init(surface: *wasi_nn_abi.Surface) Resolver {
        return .{
            .surface = surface,
            .allocator = surface.allocator,
            .memory = surface.memory,
        };
    }

    pub fn initWasi(allocator: std.mem.Allocator, memory: *wasi_nn_abi.LinearMemory) Resolver {
        return initWasiConfig(allocator, memory, .{});
    }

    pub fn initWasiConfig(
        allocator: std.mem.Allocator,
        memory: *wasi_nn_abi.LinearMemory,
        config: WasiConfig,
    ) Resolver {
        return .{
            .allocator = allocator,
            .memory = memory,
            .args = config.args,
            .environ = config.environ,
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

        if (std.mem.eql(u8, module_name, zug_nn_module_name)) {
            if (std.mem.eql(u8, function_name, "load_preloaded_graph")) return .load_preloaded_graph;

            return null;
        }

        if (std.mem.eql(u8, module_name, wasi_module_name)) {
            if (std.mem.eql(u8, function_name, "args_sizes_get")) return .args_sizes_get;
            if (std.mem.eql(u8, function_name, "args_get")) return .args_get;
            if (std.mem.eql(u8, function_name, "environ_sizes_get")) return .environ_sizes_get;
            if (std.mem.eql(u8, function_name, "environ_get")) return .environ_get;
            if (std.mem.eql(u8, function_name, "clock_time_get")) return .clock_time_get;
            if (std.mem.eql(u8, function_name, "random_get")) return .random_get;
            if (std.mem.eql(u8, function_name, "fd_fdstat_get")) return .fd_fdstat_get;
            if (std.mem.eql(u8, function_name, "fd_write")) return .fd_write;
            if (std.mem.eql(u8, function_name, "proc_exit")) return .proc_exit;

            return null;
        }

        return null;
    }

    pub fn call(self: *Resolver, function: Function, args: []const Arg) u32 {
        const status = switch (function) {
            .load_graph => blk: {
                if (args.len != 5) break :blk wasi_nn_abi.Status.runtime_error;
                const surface = self.surface orelse break :blk wasi_nn_abi.Status.runtime_error;
                break :blk surface.loadGraph(
                    argAsI32(args[0]) catch break :blk wasi_nn_abi.Status.runtime_error,
                    argAsI32(args[1]) catch break :blk wasi_nn_abi.Status.runtime_error,
                    argAsI32(args[2]) catch break :blk wasi_nn_abi.Status.runtime_error,
                    argAsI32(args[3]) catch break :blk wasi_nn_abi.Status.runtime_error,
                    argAsI32(args[4]) catch break :blk wasi_nn_abi.Status.runtime_error,
                );
            },
            .load_preloaded_graph => blk: {
                if (args.len != 3) break :blk wasi_nn_abi.Status.runtime_error;
                const surface = self.surface orelse break :blk wasi_nn_abi.Status.runtime_error;
                break :blk surface.loadPreloadedGraph(
                    argAsI32(args[0]) catch break :blk wasi_nn_abi.Status.runtime_error,
                    argAsI32(args[1]) catch break :blk wasi_nn_abi.Status.runtime_error,
                    argAsI32(args[2]) catch break :blk wasi_nn_abi.Status.runtime_error,
                );
            },
            .init_execution_context => blk: {
                if (args.len != 2) break :blk wasi_nn_abi.Status.runtime_error;
                const surface = self.surface orelse break :blk wasi_nn_abi.Status.runtime_error;
                break :blk surface.initExecutionContext(
                    argAsI32(args[0]) catch break :blk wasi_nn_abi.Status.runtime_error,
                    argAsI32(args[1]) catch break :blk wasi_nn_abi.Status.runtime_error,
                );
            },
            .set_input_by_index => blk: {
                if (args.len != 7) break :blk wasi_nn_abi.Status.runtime_error;
                const surface = self.surface orelse break :blk wasi_nn_abi.Status.runtime_error;
                break :blk surface.setInputByIndex(
                    argAsI32(args[0]) catch break :blk wasi_nn_abi.Status.runtime_error,
                    argAsI32(args[1]) catch break :blk wasi_nn_abi.Status.runtime_error,
                    argAsI32(args[2]) catch break :blk wasi_nn_abi.Status.runtime_error,
                    argAsI32(args[3]) catch break :blk wasi_nn_abi.Status.runtime_error,
                    argAsI32(args[4]) catch break :blk wasi_nn_abi.Status.runtime_error,
                    argAsI32(args[5]) catch break :blk wasi_nn_abi.Status.runtime_error,
                    argAsI32(args[6]) catch break :blk wasi_nn_abi.Status.runtime_error,
                );
            },
            .compute => blk: {
                if (args.len != 1) break :blk wasi_nn_abi.Status.runtime_error;
                const surface = self.surface orelse break :blk wasi_nn_abi.Status.runtime_error;
                break :blk surface.compute(argAsI32(args[0]) catch break :blk wasi_nn_abi.Status.runtime_error);
            },
            .get_output_descriptor => blk: {
                if (args.len != 7) break :blk wasi_nn_abi.Status.runtime_error;
                const surface = self.surface orelse break :blk wasi_nn_abi.Status.runtime_error;
                break :blk surface.getOutputDescriptor(
                    argAsI32(args[0]) catch break :blk wasi_nn_abi.Status.runtime_error,
                    argAsI32(args[1]) catch break :blk wasi_nn_abi.Status.runtime_error,
                    argAsI32(args[2]) catch break :blk wasi_nn_abi.Status.runtime_error,
                    argAsI32(args[3]) catch break :blk wasi_nn_abi.Status.runtime_error,
                    argAsI32(args[4]) catch break :blk wasi_nn_abi.Status.runtime_error,
                    argAsI32(args[5]) catch break :blk wasi_nn_abi.Status.runtime_error,
                    argAsI32(args[6]) catch break :blk wasi_nn_abi.Status.runtime_error,
                );
            },
            .get_output => blk: {
                if (args.len != 5) break :blk wasi_nn_abi.Status.runtime_error;
                const surface = self.surface orelse break :blk wasi_nn_abi.Status.runtime_error;
                break :blk surface.getOutput(
                    argAsI32(args[0]) catch break :blk wasi_nn_abi.Status.runtime_error,
                    argAsI32(args[1]) catch break :blk wasi_nn_abi.Status.runtime_error,
                    argAsI32(args[2]) catch break :blk wasi_nn_abi.Status.runtime_error,
                    argAsI32(args[3]) catch break :blk wasi_nn_abi.Status.runtime_error,
                    argAsI32(args[4]) catch break :blk wasi_nn_abi.Status.runtime_error,
                );
            },
            .args_sizes_get => return self.argsSizesGet(args),
            .args_get => return self.argsGet(args),
            .environ_sizes_get => return self.environSizesGet(args),
            .environ_get => return self.environGet(args),
            .clock_time_get => return self.clockTimeGet(args),
            .random_get => return self.randomGet(args),
            .fd_fdstat_get => return self.fdFdstatGet(args),
            .fd_write => return self.fdWrite(args),
            .proc_exit => {
                if (args.len != 1) return wasiErrnoInval;
                self.exit_code = argAsI32(args[0]) catch return wasiErrnoInval;
                return 0;
            },
        };

        return @intFromEnum(status);
    }

    fn argsSizesGet(self: *Resolver, args: []const Arg) u32 {
        if (args.len != 2) return wasiErrnoInval;
        return writeStringListSizes(self.memory orelse return wasiErrnoFault, self.args, args) catch |err| {
            return errnoFromMemoryError(err);
        };
    }

    fn argsGet(self: *Resolver, args: []const Arg) u32 {
        if (args.len != 2) return wasiErrnoInval;
        return writeStringList(self.memory orelse return wasiErrnoFault, self.args, args) catch |err| {
            return errnoFromMemoryError(err);
        };
    }

    fn environSizesGet(self: *Resolver, args: []const Arg) u32 {
        if (args.len != 2) return wasiErrnoInval;
        return writeStringListSizes(self.memory orelse return wasiErrnoFault, self.environ, args) catch |err| {
            return errnoFromMemoryError(err);
        };
    }

    fn environGet(self: *Resolver, args: []const Arg) u32 {
        if (args.len != 2) return wasiErrnoInval;
        return writeStringList(self.memory orelse return wasiErrnoFault, self.environ, args) catch |err| {
            return errnoFromMemoryError(err);
        };
    }

    fn clockTimeGet(self: *Resolver, args: []const Arg) u32 {
        if (args.len != 3) return wasiErrnoInval;

        const clock_id = argAsI32(args[0]) catch return wasiErrnoInval;
        _ = argAsI64(args[1]) catch return wasiErrnoInval;
        const time_ptr = argAsI32(args[2]) catch return wasiErrnoInval;
        const memory = self.memory orelse return wasiErrnoFault;

        switch (clock_id) {
            0, 1, 2, 3 => {},
            else => return wasiErrnoInval,
        }

        const clock: std.Io.Clock = switch (clock_id) {
            0 => .real,
            1 => .awake,
            2 => .cpu_process,
            3 => .cpu_thread,
            else => return wasiErrnoInval,
        };
        const now = clock.now(std.Options.debug_io);
        const timestamp = std.math.cast(u64, now.toNanoseconds()) orelse self.nextFallbackTimestamp();
        memory.writeU64(time_ptr, timestamp) catch return wasiErrnoFault;

        return wasiErrnoSuccess;
    }

    fn randomGet(self: *Resolver, args: []const Arg) u32 {
        if (args.len != 2) return wasiErrnoInval;

        const buf_ptr = argAsI32(args[0]) catch return wasiErrnoInval;
        const buf_len = argAsI32(args[1]) catch return wasiErrnoInval;
        const memory = self.memory orelse return wasiErrnoFault;
        const bytes = memory.writeSlice(buf_ptr, buf_len) catch return wasiErrnoFault;

        std.Io.randomSecure(std.Options.debug_io, bytes) catch {
            std.Io.random(std.Options.debug_io, bytes);
        };
        return wasiErrnoSuccess;
    }

    fn fdFdstatGet(self: *Resolver, args: []const Arg) u32 {
        if (args.len != 2) return wasiErrnoInval;

        const fd = argAsI32(args[0]) catch return wasiErrnoInval;
        const stat_ptr = argAsI32(args[1]) catch return wasiErrnoInval;
        const memory = self.memory orelse return wasiErrnoFault;

        switch (fd) {
            0, 1, 2 => {},
            else => return wasiErrnoBadf,
        }

        var stat = [_]u8{0} ** 24;
        stat[0] = 2;
        writeU16Little(stat[2..4], 0);
        writeU64Little(stat[8..16], std.math.maxInt(u64));
        writeU64Little(stat[16..24], std.math.maxInt(u64));
        memory.write(stat_ptr, &stat) catch return wasiErrnoFault;

        return wasiErrnoSuccess;
    }

    fn fdWrite(self: *Resolver, args: []const Arg) u32 {
        if (args.len != 4) return wasiErrnoInval;

        const allocator = self.allocator orelse return wasiErrnoInval;
        const memory = self.memory orelse return wasiErrnoFault;
        const fd = argAsI32(args[0]) catch return wasiErrnoInval;
        const iovs_ptr = argAsI32(args[1]) catch return wasiErrnoInval;
        const iovs_len = argAsI32(args[2]) catch return wasiErrnoInval;
        const nwritten_ptr = argAsI32(args[3]) catch return wasiErrnoInval;

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

    fn nextFallbackTimestamp(self: *Resolver) u64 {
        const current = self.fallback_clock_ns;
        self.fallback_clock_ns +|= 1_000_000;
        return current;
    }
};

fn argAsI32(arg: Arg) !u32 {
    return switch (arg) {
        .i32 => |value| value,
        else => error.InvalidWasiArgType,
    };
}

fn argAsI64(arg: Arg) !u64 {
    return switch (arg) {
        .i64 => |value| value,
        else => error.InvalidWasiArgType,
    };
}

fn writeStringListSizes(
    memory: *wasi_nn_abi.LinearMemory,
    values: []const []const u8,
    args: []const Arg,
) !u32 {
    const count_ptr = try argAsI32(args[0]);
    const byte_count_ptr = try argAsI32(args[1]);
    const count = std.math.cast(u32, values.len) orelse return error.InvalidWasiStringList;
    const byte_count = try stringListByteLen(values);

    try memory.writeU32(count_ptr, count);
    try memory.writeU32(byte_count_ptr, byte_count);

    return wasiErrnoSuccess;
}

fn writeStringList(
    memory: *wasi_nn_abi.LinearMemory,
    values: []const []const u8,
    args: []const Arg,
) !u32 {
    const ptrs_ptr = try argAsI32(args[0]);
    var bytes_ptr = try argAsI32(args[1]);

    for (values, 0..) |value, index| {
        const index_u32 = std.math.cast(u32, index) orelse return error.InvalidWasiStringList;
        const slot_ptr = try std.math.add(u32, ptrs_ptr, index_u32 * 4);
        try memory.writeU32(slot_ptr, bytes_ptr);
        try memory.write(bytes_ptr, value);

        const terminator = [_]u8{0};
        bytes_ptr = try std.math.add(u32, bytes_ptr, std.math.cast(u32, value.len) orelse {
            return error.InvalidWasiStringList;
        });
        try memory.write(bytes_ptr, &terminator);
        bytes_ptr = try std.math.add(u32, bytes_ptr, 1);
    }

    return wasiErrnoSuccess;
}

fn stringListByteLen(values: []const []const u8) !u32 {
    var byte_count: u32 = 0;
    for (values) |value| {
        const item_len = std.math.cast(u32, value.len) orelse return error.InvalidWasiStringList;
        byte_count = try std.math.add(u32, byte_count, item_len);
        byte_count = try std.math.add(u32, byte_count, 1);
    }
    return byte_count;
}

fn errnoFromMemoryError(err: anyerror) u32 {
    return switch (err) {
        error.InvalidMemoryRange,
        error.Overflow,
        => wasiErrnoFault,
        error.InvalidWasiArgType,
        error.InvalidWasiStringList,
        => wasiErrnoInval,
        else => wasiErrnoIo,
    };
}

fn writeU16Little(bytes: []u8, value: u16) void {
    bytes[0] = std.math.cast(u8, value & 0xff) orelse unreachable;
    bytes[1] = std.math.cast(u8, (value >> 8) & 0xff) orelse unreachable;
}

fn writeU64Little(bytes: []u8, value: u64) void {
    bytes[0] = std.math.cast(u8, value & 0xff) orelse unreachable;
    bytes[1] = std.math.cast(u8, (value >> 8) & 0xff) orelse unreachable;
    bytes[2] = std.math.cast(u8, (value >> 16) & 0xff) orelse unreachable;
    bytes[3] = std.math.cast(u8, (value >> 24) & 0xff) orelse unreachable;
    bytes[4] = std.math.cast(u8, (value >> 32) & 0xff) orelse unreachable;
    bytes[5] = std.math.cast(u8, (value >> 40) & 0xff) orelse unreachable;
    bytes[6] = std.math.cast(u8, (value >> 48) & 0xff) orelse unreachable;
    bytes[7] = std.math.cast(u8, (value >> 56) & 0xff) orelse unreachable;
}

const wasiErrnoSuccess = 0;
const wasiErrnoBadf = 8;
const wasiErrnoFault = 21;
const wasiErrnoInval = 28;
const wasiErrnoIo = 29;
const wasiErrnoOverflow = 61;

test "resolver maps wasi-nn import names" {
    try std.testing.expectEqual(Function.load_graph, Resolver.resolve("wasi_nn", "load_graph").?);
    try std.testing.expectEqual(Function.compute, Resolver.resolve("wasi_nn", "compute").?);
    try std.testing.expectEqual(Function.load_preloaded_graph, Resolver.resolve("zug_nn", "load_preloaded_graph").?);
    try std.testing.expect(Resolver.resolve("env", "compute") == null);
}

test "resolver maps wasi imports" {
    try std.testing.expectEqual(Function.args_sizes_get, Resolver.resolve(wasi_module_name, "args_sizes_get").?);
    try std.testing.expectEqual(Function.args_get, Resolver.resolve(wasi_module_name, "args_get").?);
    try std.testing.expectEqual(Function.environ_sizes_get, Resolver.resolve(wasi_module_name, "environ_sizes_get").?);
    try std.testing.expectEqual(Function.environ_get, Resolver.resolve(wasi_module_name, "environ_get").?);
    try std.testing.expectEqual(Function.clock_time_get, Resolver.resolve(wasi_module_name, "clock_time_get").?);
    try std.testing.expectEqual(Function.random_get, Resolver.resolve(wasi_module_name, "random_get").?);
    try std.testing.expectEqual(Function.fd_fdstat_get, Resolver.resolve(wasi_module_name, "fd_fdstat_get").?);
    try std.testing.expectEqual(Function.fd_write, Resolver.resolve(wasi_module_name, "fd_write").?);
    try std.testing.expectEqual(Function.proc_exit, Resolver.resolve(wasi_module_name, "proc_exit").?);
}

test "resolver records proc_exit code" {
    var memory_bytes = [_]u8{0} ** 64;
    var memory = wasi_nn_abi.LinearMemory.init(&memory_bytes);
    var resolver = Resolver.initWasi(std.testing.allocator, &memory);
    defer resolver.deinit();

    try std.testing.expectEqual(@as(u32, 0), resolver.call(.proc_exit, &.{.{ .i32 = 7 }}));
    try std.testing.expectEqual(@as(?u32, 7), resolver.exit_code);
}

test "resolver exposes wasi args and environ" {
    var memory_bytes = [_]u8{0} ** 128;
    var memory = wasi_nn_abi.LinearMemory.init(&memory_bytes);
    const args = [_][]const u8{ "zug", "run" };
    const environ = [_][]const u8{"ZUG_EDGE=1"};
    var resolver = Resolver.initWasiConfig(std.testing.allocator, &memory, .{
        .args = &args,
        .environ = &environ,
    });
    defer resolver.deinit();

    try std.testing.expectEqual(
        wasiErrnoSuccess,
        resolver.call(.args_sizes_get, &.{ .{ .i32 = 0 }, .{ .i32 = 4 } }),
    );
    try std.testing.expectEqual(@as(u32, 2), try memory.readU32(0));
    try std.testing.expectEqual(@as(u32, 8), try memory.readU32(4));

    try std.testing.expectEqual(
        wasiErrnoSuccess,
        resolver.call(.args_get, &.{ .{ .i32 = 16 }, .{ .i32 = 32 } }),
    );
    try std.testing.expectEqual(@as(u32, 32), try memory.readU32(16));
    try std.testing.expectEqual(@as(u32, 36), try memory.readU32(20));
    try std.testing.expectEqualStrings("zug\x00run\x00", try memory.read(32, 8));

    try std.testing.expectEqual(
        wasiErrnoSuccess,
        resolver.call(.environ_sizes_get, &.{ .{ .i32 = 48 }, .{ .i32 = 52 } }),
    );
    try std.testing.expectEqual(@as(u32, 1), try memory.readU32(48));
    try std.testing.expectEqual(@as(u32, 11), try memory.readU32(52));
}

test "resolver exposes wasi clock random and fdstat" {
    var memory_bytes = [_]u8{0} ** 128;
    var memory = wasi_nn_abi.LinearMemory.init(&memory_bytes);
    var resolver = Resolver.initWasi(std.testing.allocator, &memory);
    defer resolver.deinit();

    try std.testing.expectEqual(
        wasiErrnoSuccess,
        resolver.call(.clock_time_get, &.{ .{ .i32 = 0 }, .{ .i64 = 0 }, .{ .i32 = 0 } }),
    );
    try std.testing.expect((try memory.readU32(0)) != 0);

    try std.testing.expectEqual(
        wasiErrnoSuccess,
        resolver.call(.random_get, &.{ .{ .i32 = 16 }, .{ .i32 = 8 } }),
    );

    try std.testing.expectEqual(
        wasiErrnoSuccess,
        resolver.call(.fd_fdstat_get, &.{ .{ .i32 = 1 }, .{ .i32 = 32 } }),
    );
    try std.testing.expectEqual(@as(u8, 2), (try memory.read(32, 1))[0]);
}
