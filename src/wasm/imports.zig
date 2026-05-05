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
    fd_close,
    fd_fdstat_get,
    fd_filestat_get,
    fd_prestat_get,
    fd_prestat_dir_name,
    fd_read,
    fd_readdir,
    fd_seek,
    fd_write,
    path_filestat_get,
    path_open,
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
    stdin: []const u8 = &.{},
    preopens: []const Preopen = &.{},
    readonly_files: []const ReadonlyFile = &.{},
};

pub const Preopen = struct {
    name: []const u8,
    allow_readdir: bool = true,
    allow_path_open: bool = true,
    allow_path_stat: bool = true,
};

pub const ReadonlyFile = struct {
    preopen_name: []const u8,
    path: []const u8,
    data: []const u8,
};

const OpenFile = struct {
    file_index: usize,
    offset: usize = 0,
    closed: bool = false,
};

const DirectoryEntry = struct {
    name: []const u8,
    filetype: u8,
};

pub const Resolver = struct {
    surface: ?*wasi_nn_abi.Surface = null,
    allocator: ?std.mem.Allocator = null,
    memory: ?*wasi_nn_abi.LinearMemory = null,
    args: []const []const u8 = &.{},
    environ: []const []const u8 = &.{},
    stdin: []const u8 = &.{},
    stdin_offset: usize = 0,
    preopens: []const Preopen = &.{},
    readonly_files: []const ReadonlyFile = &.{},
    open_files: std.ArrayList(OpenFile) = .empty,
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
            .stdin = config.stdin,
            .preopens = config.preopens,
            .readonly_files = config.readonly_files,
        };
    }

    pub fn deinit(self: *Resolver) void {
        if (self.allocator) |allocator| {
            self.open_files.deinit(allocator);
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
            if (std.mem.eql(u8, function_name, "fd_close")) return .fd_close;
            if (std.mem.eql(u8, function_name, "fd_fdstat_get")) return .fd_fdstat_get;
            if (std.mem.eql(u8, function_name, "fd_filestat_get")) return .fd_filestat_get;
            if (std.mem.eql(u8, function_name, "fd_prestat_get")) return .fd_prestat_get;
            if (std.mem.eql(u8, function_name, "fd_prestat_dir_name")) return .fd_prestat_dir_name;
            if (std.mem.eql(u8, function_name, "fd_read")) return .fd_read;
            if (std.mem.eql(u8, function_name, "fd_readdir")) return .fd_readdir;
            if (std.mem.eql(u8, function_name, "fd_seek")) return .fd_seek;
            if (std.mem.eql(u8, function_name, "fd_write")) return .fd_write;
            if (std.mem.eql(u8, function_name, "path_filestat_get")) return .path_filestat_get;
            if (std.mem.eql(u8, function_name, "path_open")) return .path_open;
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
            .fd_close => return self.fdClose(args),
            .fd_fdstat_get => return self.fdFdstatGet(args),
            .fd_filestat_get => return self.fdFilestatGet(args),
            .fd_prestat_get => return self.fdPrestatGet(args),
            .fd_prestat_dir_name => return self.fdPrestatDirName(args),
            .fd_read => return self.fdRead(args),
            .fd_readdir => return self.fdReaddir(args),
            .fd_seek => return self.fdSeek(args),
            .fd_write => return self.fdWrite(args),
            .path_filestat_get => return self.pathFilestatGet(args),
            .path_open => return self.pathOpen(args),
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

    fn fdClose(self: *Resolver, args: []const Arg) u32 {
        if (args.len != 1) return wasiErrnoInval;

        const fd = argAsI32(args[0]) catch return wasiErrnoInval;
        if (fd <= 2) return wasiErrnoSuccess;
        if (self.preopenIndexForFd(fd) != null) return wasiErrnoSuccess;

        const open_file = self.openFileForFd(fd) orelse return wasiErrnoBadf;
        if (open_file.closed) return wasiErrnoBadf;
        open_file.closed = true;
        return wasiErrnoSuccess;
    }

    fn fdFdstatGet(self: *Resolver, args: []const Arg) u32 {
        if (args.len != 2) return wasiErrnoInval;

        const fd = argAsI32(args[0]) catch return wasiErrnoInval;
        const stat_ptr = argAsI32(args[1]) catch return wasiErrnoInval;
        const memory = self.memory orelse return wasiErrnoFault;
        const filetype = self.fileTypeForFd(fd) orelse return wasiErrnoBadf;

        const rights = self.rightsForFd(fd) orelse return wasiErrnoBadf;
        writeFdstat(memory, stat_ptr, filetype, rights) catch return wasiErrnoFault;

        return wasiErrnoSuccess;
    }

    fn fdFilestatGet(self: *Resolver, args: []const Arg) u32 {
        if (args.len != 2) return wasiErrnoInval;

        const fd = argAsI32(args[0]) catch return wasiErrnoInval;
        const stat_ptr = argAsI32(args[1]) catch return wasiErrnoInval;
        const memory = self.memory orelse return wasiErrnoFault;

        if (fd <= 2) {
            writeFilestat(memory, stat_ptr, wasiFiletypeCharacterDevice, 0) catch return wasiErrnoFault;
            return wasiErrnoSuccess;
        }

        if (self.preopenIndexForFd(fd)) |_| {
            writeFilestat(memory, stat_ptr, wasiFiletypeDirectory, 0) catch return wasiErrnoFault;
            return wasiErrnoSuccess;
        }

        const open_file = self.openFileForFd(fd) orelse return wasiErrnoBadf;
        if (open_file.closed) return wasiErrnoBadf;
        const file = self.readonly_files[open_file.file_index];
        writeFilestat(memory, stat_ptr, wasiFiletypeRegularFile, file.data.len) catch return wasiErrnoFault;

        return wasiErrnoSuccess;
    }

    fn fdPrestatGet(self: *Resolver, args: []const Arg) u32 {
        if (args.len != 2) return wasiErrnoInval;

        const fd = argAsI32(args[0]) catch return wasiErrnoInval;
        const prestat_ptr = argAsI32(args[1]) catch return wasiErrnoInval;
        const memory = self.memory orelse return wasiErrnoFault;
        const preopen_index = self.preopenIndexForFd(fd) orelse return wasiErrnoBadf;
        const name = self.preopens[preopen_index].name;

        var prestat = [_]u8{0} ** 8;
        prestat[0] = 0;
        writeU32Little(prestat[4..8], std.math.cast(u32, name.len) orelse return wasiErrnoOverflow);
        memory.write(prestat_ptr, &prestat) catch return wasiErrnoFault;

        return wasiErrnoSuccess;
    }

    fn fdPrestatDirName(self: *Resolver, args: []const Arg) u32 {
        if (args.len != 3) return wasiErrnoInval;

        const fd = argAsI32(args[0]) catch return wasiErrnoInval;
        const path_ptr = argAsI32(args[1]) catch return wasiErrnoInval;
        const path_len = argAsI32(args[2]) catch return wasiErrnoInval;
        const memory = self.memory orelse return wasiErrnoFault;
        const preopen_index = self.preopenIndexForFd(fd) orelse return wasiErrnoBadf;
        const name = self.preopens[preopen_index].name;

        const target_len = std.math.cast(usize, path_len) orelse return wasiErrnoInval;
        if (target_len < name.len) return wasiErrnoNametoolong;

        memory.write(path_ptr, name) catch return wasiErrnoFault;
        return wasiErrnoSuccess;
    }

    fn fdRead(self: *Resolver, args: []const Arg) u32 {
        if (args.len != 4) return wasiErrnoInval;

        const fd = argAsI32(args[0]) catch return wasiErrnoInval;
        const iovs_ptr = argAsI32(args[1]) catch return wasiErrnoInval;
        const iovs_len = argAsI32(args[2]) catch return wasiErrnoInval;
        const nread_ptr = argAsI32(args[3]) catch return wasiErrnoInval;
        const memory = self.memory orelse return wasiErrnoFault;

        if (fd == 0) {
            return readSourceIntoIovs(memory, self.stdin, &self.stdin_offset, iovs_ptr, iovs_len, nread_ptr);
        }

        const open_file = self.openFileForFd(fd) orelse return wasiErrnoBadf;
        if (open_file.closed) return wasiErrnoBadf;
        const file = self.readonly_files[open_file.file_index];
        return readSourceIntoIovs(memory, file.data, &open_file.offset, iovs_ptr, iovs_len, nread_ptr);
    }

    fn fdReaddir(self: *Resolver, args: []const Arg) u32 {
        if (args.len != 5) return wasiErrnoInval;

        const allocator = self.allocator orelse return wasiErrnoInval;
        const fd = argAsI32(args[0]) catch return wasiErrnoInval;
        const buf_ptr = argAsI32(args[1]) catch return wasiErrnoInval;
        const buf_len = argAsI32(args[2]) catch return wasiErrnoInval;
        const cookie = argAsI64(args[3]) catch return wasiErrnoInval;
        const bufused_ptr = argAsI32(args[4]) catch return wasiErrnoInval;
        const memory = self.memory orelse return wasiErrnoFault;
        const preopen_index = self.preopenIndexForFd(fd) orelse return wasiErrnoNotdir;
        const preopen = self.preopens[preopen_index];
        if (!preopen.allow_readdir) return wasiErrnoAcces;

        var entries: std.ArrayList(DirectoryEntry) = .empty;
        defer entries.deinit(allocator);
        appendDirectoryEntries(allocator, &entries, preopen.name, "", self.readonly_files) catch return wasiErrnoNomem;

        var used: u32 = 0;
        const start_index = std.math.cast(usize, cookie) orelse return wasiErrnoInval;
        if (start_index > entries.items.len) {
            memory.writeU32(bufused_ptr, 0) catch return wasiErrnoFault;
            return wasiErrnoSuccess;
        }

        const max_len = std.math.cast(usize, buf_len) orelse return wasiErrnoInval;
        for (entries.items[start_index..], start_index..) |entry, index| {
            const entry_size = 24 + entry.name.len;
            const used_usize = std.math.cast(usize, used) orelse return wasiErrnoInval;
            if (entry_size > max_len -| used_usize) break;

            const entry_ptr = std.math.add(u32, buf_ptr, used) catch return wasiErrnoFault;
            writeDirent(memory, entry_ptr, index + 1, entry) catch return wasiErrnoFault;
            const used_delta = std.math.cast(u32, entry_size) orelse return wasiErrnoOverflow;
            used = std.math.add(u32, used, used_delta) catch return wasiErrnoOverflow;
        }

        memory.writeU32(bufused_ptr, used) catch return wasiErrnoFault;
        return wasiErrnoSuccess;
    }

    fn fdSeek(self: *Resolver, args: []const Arg) u32 {
        if (args.len != 4) return wasiErrnoInval;

        const fd = argAsI32(args[0]) catch return wasiErrnoInval;
        const offset_raw = argAsI64(args[1]) catch return wasiErrnoInval;
        const whence = argAsI32(args[2]) catch return wasiErrnoInval;
        const newoffset_ptr = argAsI32(args[3]) catch return wasiErrnoInval;
        const memory = self.memory orelse return wasiErrnoFault;
        const open_file = self.openFileForFd(fd) orelse return wasiErrnoBadf;
        if (open_file.closed) return wasiErrnoBadf;

        const file = self.readonly_files[open_file.file_index];
        const offset: i64 = @bitCast(offset_raw);
        const base: i128 = switch (whence) {
            0 => 0,
            1 => std.math.cast(i128, open_file.offset) orelse return wasiErrnoOverflow,
            2 => std.math.cast(i128, file.data.len) orelse return wasiErrnoOverflow,
            else => return wasiErrnoInval,
        };
        const next = base + offset;
        if (next < 0) return wasiErrnoInval;

        open_file.offset = std.math.cast(usize, next) orelse return wasiErrnoOverflow;
        memory.writeU64(newoffset_ptr, std.math.cast(u64, open_file.offset) orelse return wasiErrnoOverflow) catch return wasiErrnoFault;
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

    fn pathFilestatGet(self: *Resolver, args: []const Arg) u32 {
        if (args.len != 5) return wasiErrnoInval;

        const dirfd = argAsI32(args[0]) catch return wasiErrnoInval;
        _ = argAsI32(args[1]) catch return wasiErrnoInval;
        const path_ptr = argAsI32(args[2]) catch return wasiErrnoInval;
        const path_len = argAsI32(args[3]) catch return wasiErrnoInval;
        const stat_ptr = argAsI32(args[4]) catch return wasiErrnoInval;
        const memory = self.memory orelse return wasiErrnoFault;
        const preopen_index = self.preopenIndexForFd(dirfd) orelse return wasiErrnoBadf;
        const preopen = self.preopens[preopen_index];
        if (!preopen.allow_path_stat) return wasiErrnoAcces;
        const path = readGuestPath(memory, path_ptr, path_len) catch return wasiErrnoFault;

        if (path.len == 0) {
            writeFilestat(memory, stat_ptr, wasiFiletypeDirectory, 0) catch return wasiErrnoFault;
            return wasiErrnoSuccess;
        }

        if (!isSafeRelativePath(path)) return wasiErrnoAcces;
        if (self.directoryExists(preopen.name, path)) {
            writeFilestat(memory, stat_ptr, wasiFiletypeDirectory, 0) catch return wasiErrnoFault;
            return wasiErrnoSuccess;
        }
        const file_index = self.findReadonlyFile(preopen.name, path) orelse return wasiErrnoNoent;
        const file = self.readonly_files[file_index];
        writeFilestat(memory, stat_ptr, wasiFiletypeRegularFile, file.data.len) catch return wasiErrnoFault;
        return wasiErrnoSuccess;
    }

    fn pathOpen(self: *Resolver, args: []const Arg) u32 {
        if (args.len != 9) return wasiErrnoInval;

        const allocator = self.allocator orelse return wasiErrnoInval;
        const dirfd = argAsI32(args[0]) catch return wasiErrnoInval;
        _ = argAsI32(args[1]) catch return wasiErrnoInval;
        const path_ptr = argAsI32(args[2]) catch return wasiErrnoInval;
        const path_len = argAsI32(args[3]) catch return wasiErrnoInval;
        const oflags = argAsI32(args[4]) catch return wasiErrnoInval;
        const rights_base = argAsI64(args[5]) catch return wasiErrnoInval;
        const rights_inheriting = argAsI64(args[6]) catch return wasiErrnoInval;
        const fdflags = argAsI32(args[7]) catch return wasiErrnoInval;
        const opened_fd_ptr = argAsI32(args[8]) catch return wasiErrnoInval;
        const memory = self.memory orelse return wasiErrnoFault;
        const preopen_index = self.preopenIndexForFd(dirfd) orelse return wasiErrnoBadf;
        const preopen = self.preopens[preopen_index];
        if (!preopen.allow_path_open) return wasiErrnoAcces;

        if (oflags != 0 or fdflags != 0) return wasiErrnoAcces;
        if (!rightsAreReadonly(rights_base) or
            !rightsAreReadonly(rights_inheriting))
        {
            return wasiErrnoAcces;
        }

        const path = readGuestPath(memory, path_ptr, path_len) catch return wasiErrnoFault;
        if (!isSafeRelativePath(path)) return wasiErrnoAcces;

        const file_index = self.findReadonlyFile(preopen.name, path) orelse return wasiErrnoNoent;
        tryAppendOpenFile(&self.open_files, allocator, .{ .file_index = file_index }) catch return wasiErrnoNomem;
        const fd = openFdForIndex(self.preopens.len, self.open_files.items.len - 1) catch return wasiErrnoOverflow;
        memory.writeU32(opened_fd_ptr, fd) catch return wasiErrnoFault;

        return wasiErrnoSuccess;
    }

    fn nextFallbackTimestamp(self: *Resolver) u64 {
        const current = self.fallback_clock_ns;
        self.fallback_clock_ns +|= 1_000_000;
        return current;
    }

    fn preopenIndexForFd(self: *const Resolver, fd: u32) ?usize {
        if (fd < 3) return null;
        const index = std.math.cast(usize, fd - 3) orelse return null;
        if (index >= self.preopens.len) return null;
        return index;
    }

    fn openFileForFd(self: *Resolver, fd: u32) ?*OpenFile {
        const first_open_fd = firstOpenFd(self.preopens.len) catch return null;
        if (fd < first_open_fd) return null;
        const index = std.math.cast(usize, fd - first_open_fd) orelse return null;
        if (index >= self.open_files.items.len) return null;
        return &self.open_files.items[index];
    }

    fn fileTypeForFd(self: *Resolver, fd: u32) ?u8 {
        if (fd <= 2) return wasiFiletypeCharacterDevice;
        if (self.preopenIndexForFd(fd) != null) return wasiFiletypeDirectory;
        const open_file = self.openFileForFd(fd) orelse return null;
        if (open_file.closed) return null;
        return wasiFiletypeRegularFile;
    }

    fn rightsForFd(self: *Resolver, fd: u32) ?u64 {
        if (fd == 0) return wasiRightFdRead;
        if (fd == 1 or fd == 2) return wasiRightFdWrite;
        if (self.preopenIndexForFd(fd) != null) return wasiDirectoryRights;

        const open_file = self.openFileForFd(fd) orelse return null;
        if (open_file.closed) return null;
        return wasiReadonlyFileRights;
    }

    fn findReadonlyFile(self: *const Resolver, preopen_name: []const u8, raw_path: []const u8) ?usize {
        const path = trimRelativePrefix(raw_path);
        for (self.readonly_files, 0..) |file, index| {
            if (std.mem.eql(u8, file.preopen_name, preopen_name) and
                std.mem.eql(u8, file.path, path))
            {
                return index;
            }
        }

        return null;
    }

    fn directoryExists(self: *const Resolver, preopen_name: []const u8, raw_path: []const u8) bool {
        const path = trimRelativePrefix(raw_path);
        for (self.readonly_files) |file| {
            if (!std.mem.eql(u8, file.preopen_name, preopen_name)) continue;
            if (file.path.len <= path.len) continue;
            if (!std.mem.eql(u8, file.path[0..path.len], path)) continue;
            if (file.path[path.len] == '/' or file.path[path.len] == '\\') return true;
        }

        return false;
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

fn readSourceIntoIovs(
    memory: *wasi_nn_abi.LinearMemory,
    source: []const u8,
    source_offset: *usize,
    iovs_ptr: u32,
    iovs_len: u32,
    nread_ptr: u32,
) u32 {
    const iovs_count = std.math.cast(usize, iovs_len) orelse return wasiErrnoInval;
    var read_total: u32 = 0;

    for (0..iovs_count) |index| {
        const index_u32 = std.math.cast(u32, index) orelse return wasiErrnoInval;
        const iov_ptr = std.math.add(u32, iovs_ptr, index_u32 * 8) catch return wasiErrnoFault;
        const buf_ptr = memory.readU32(iov_ptr) catch return wasiErrnoFault;
        const buf_len = memory.readU32(iov_ptr + 4) catch return wasiErrnoFault;
        const dest_len = std.math.cast(usize, buf_len) orelse return wasiErrnoInval;

        if (source_offset.* >= source.len) break;

        const available = source[source_offset.*..];
        const copy_len = @min(dest_len, available.len);
        memory.write(buf_ptr, available[0..copy_len]) catch return wasiErrnoFault;
        source_offset.* += copy_len;
        read_total = std.math.add(u32, read_total, std.math.cast(u32, copy_len) orelse return wasiErrnoOverflow) catch {
            return wasiErrnoOverflow;
        };

        if (copy_len < dest_len) break;
    }

    memory.writeU32(nread_ptr, read_total) catch return wasiErrnoFault;
    return wasiErrnoSuccess;
}

fn readGuestPath(memory: *wasi_nn_abi.LinearMemory, path_ptr: u32, path_len: u32) ![]const u8 {
    return try memory.read(path_ptr, path_len);
}

fn isSafeRelativePath(raw_path: []const u8) bool {
    const path = trimRelativePrefix(raw_path);
    if (path.len == 0) return false;
    if (path[0] == '/' or path[0] == '\\') return false;

    var parts = std.mem.splitAny(u8, path, "/\\");
    while (parts.next()) |part| {
        if (part.len == 0) return false;
        if (std.mem.eql(u8, part, "..")) return false;
    }

    return true;
}

fn trimRelativePrefix(path: []const u8) []const u8 {
    var actual = path;
    while (std.mem.startsWith(u8, actual, "./") or std.mem.startsWith(u8, actual, ".\\")) {
        actual = actual[2..];
    }
    return actual;
}

fn tryAppendOpenFile(
    open_files: *std.ArrayList(OpenFile),
    allocator: std.mem.Allocator,
    open_file: OpenFile,
) !void {
    try open_files.append(allocator, open_file);
}

fn firstOpenFd(preopen_count: usize) !u32 {
    return try std.math.add(u32, 3, std.math.cast(u32, preopen_count) orelse return error.Overflow);
}

fn openFdForIndex(preopen_count: usize, open_file_index: usize) !u32 {
    return try std.math.add(u32, try firstOpenFd(preopen_count), std.math.cast(u32, open_file_index) orelse {
        return error.Overflow;
    });
}

fn appendDirectoryEntries(
    allocator: std.mem.Allocator,
    entries: *std.ArrayList(DirectoryEntry),
    preopen_name: []const u8,
    dir_path_raw: []const u8,
    files: []const ReadonlyFile,
) !void {
    const dir_path = trimRelativePrefix(dir_path_raw);

    for (files) |file| {
        if (!std.mem.eql(u8, file.preopen_name, preopen_name)) continue;
        const child = immediateDirectoryChild(file.path, dir_path) orelse continue;
        if (hasDirectoryEntry(entries.items, child.name)) continue;

        try entries.append(allocator, .{
            .name = child.name,
            .filetype = if (child.is_directory) wasiFiletypeDirectory else wasiFiletypeRegularFile,
        });
    }
}

const DirectoryChild = struct {
    name: []const u8,
    is_directory: bool,
};

fn immediateDirectoryChild(file_path_raw: []const u8, dir_path_raw: []const u8) ?DirectoryChild {
    const file_path = trimRelativePrefix(file_path_raw);
    const dir_path = trimRelativePrefix(dir_path_raw);
    const remainder = if (dir_path.len == 0) blk: {
        break :blk file_path;
    } else blk: {
        if (file_path.len <= dir_path.len) return null;
        if (!std.mem.eql(u8, file_path[0..dir_path.len], dir_path)) return null;
        if (file_path[dir_path.len] != '/' and file_path[dir_path.len] != '\\') return null;
        break :blk file_path[dir_path.len + 1 ..];
    };

    if (remainder.len == 0) return null;
    const separator = std.mem.indexOfAny(u8, remainder, "/\\") orelse {
        return .{ .name = remainder, .is_directory = false };
    };
    if (separator == 0) return null;
    return .{ .name = remainder[0..separator], .is_directory = true };
}

fn hasDirectoryEntry(entries: []const DirectoryEntry, name: []const u8) bool {
    for (entries) |entry| {
        if (std.mem.eql(u8, entry.name, name)) return true;
    }
    return false;
}

fn writeDirent(
    memory: *wasi_nn_abi.LinearMemory,
    ptr: u32,
    next_cookie: usize,
    entry: DirectoryEntry,
) !void {
    var dirent = [_]u8{0} ** 24;
    writeU64Little(dirent[0..8], std.math.cast(u64, next_cookie) orelse return error.Overflow);
    writeU64Little(dirent[8..16], std.math.cast(u64, next_cookie) orelse return error.Overflow);
    writeU32Little(dirent[16..20], std.math.cast(u32, entry.name.len) orelse return error.Overflow);
    dirent[20] = entry.filetype;
    try memory.write(ptr, &dirent);
    try memory.write(try std.math.add(u32, ptr, 24), entry.name);
}

fn rightsAreReadonly(rights: u64) bool {
    return (rights & ~wasiReadonlyAllowedRights) == 0;
}

fn writeFdstat(memory: *wasi_nn_abi.LinearMemory, stat_ptr: u32, filetype: u8, rights: u64) !void {
    var stat = [_]u8{0} ** 24;
    stat[0] = filetype;
    writeU16Little(stat[2..4], 0);
    writeU64Little(stat[8..16], rights);
    writeU64Little(stat[16..24], rights);
    try memory.write(stat_ptr, &stat);
}

fn writeFilestat(
    memory: *wasi_nn_abi.LinearMemory,
    stat_ptr: u32,
    filetype: u8,
    size: usize,
) !void {
    var stat = [_]u8{0} ** 64;
    stat[16] = filetype;
    writeU64Little(stat[24..32], 1);
    writeU64Little(stat[32..40], std.math.cast(u64, size) orelse return error.Overflow);
    try memory.write(stat_ptr, &stat);
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

fn writeU32Little(bytes: []u8, value: u32) void {
    bytes[0] = std.math.cast(u8, value & 0xff) orelse unreachable;
    bytes[1] = std.math.cast(u8, (value >> 8) & 0xff) orelse unreachable;
    bytes[2] = std.math.cast(u8, (value >> 16) & 0xff) orelse unreachable;
    bytes[3] = std.math.cast(u8, (value >> 24) & 0xff) orelse unreachable;
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

fn readU64Little(bytes: []const u8) u64 {
    return @as(u64, bytes[0]) |
        (@as(u64, bytes[1]) << 8) |
        (@as(u64, bytes[2]) << 16) |
        (@as(u64, bytes[3]) << 24) |
        (@as(u64, bytes[4]) << 32) |
        (@as(u64, bytes[5]) << 40) |
        (@as(u64, bytes[6]) << 48) |
        (@as(u64, bytes[7]) << 56);
}

const wasiErrnoSuccess = 0;
const wasiErrnoAcces = 2;
const wasiErrnoBadf = 8;
const wasiErrnoFault = 21;
const wasiErrnoInval = 28;
const wasiErrnoIo = 29;
const wasiErrnoNametoolong = 37;
const wasiErrnoNoent = 44;
const wasiErrnoNomem = 48;
const wasiErrnoNotdir = 54;
const wasiErrnoOverflow = 61;

const wasiFiletypeCharacterDevice = 2;
const wasiFiletypeDirectory = 3;
const wasiFiletypeRegularFile = 4;

const wasiRightFdRead = @as(u64, 1) << 1;
const wasiRightFdSeek = @as(u64, 1) << 2;
const wasiRightFdWrite = @as(u64, 1) << 6;
const wasiRightPathOpen = @as(u64, 1) << 13;
const wasiRightFdReaddir = @as(u64, 1) << 14;
const wasiRightPathFilestatGet = @as(u64, 1) << 18;
const wasiRightFdFilestatGet = @as(u64, 1) << 21;

const wasiReadonlyFileRights = wasiRightFdRead |
    wasiRightFdSeek |
    wasiRightFdFilestatGet;

const wasiDirectoryRights = wasiRightFdReaddir |
    wasiRightPathOpen |
    wasiRightPathFilestatGet |
    wasiRightFdFilestatGet;

const wasiReadonlyAllowedRights = wasiReadonlyFileRights | wasiDirectoryRights;

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
    try std.testing.expectEqual(Function.fd_close, Resolver.resolve(wasi_module_name, "fd_close").?);
    try std.testing.expectEqual(Function.fd_fdstat_get, Resolver.resolve(wasi_module_name, "fd_fdstat_get").?);
    try std.testing.expectEqual(Function.fd_filestat_get, Resolver.resolve(wasi_module_name, "fd_filestat_get").?);
    try std.testing.expectEqual(Function.fd_prestat_get, Resolver.resolve(wasi_module_name, "fd_prestat_get").?);
    try std.testing.expectEqual(Function.fd_prestat_dir_name, Resolver.resolve(wasi_module_name, "fd_prestat_dir_name").?);
    try std.testing.expectEqual(Function.fd_read, Resolver.resolve(wasi_module_name, "fd_read").?);
    try std.testing.expectEqual(Function.fd_readdir, Resolver.resolve(wasi_module_name, "fd_readdir").?);
    try std.testing.expectEqual(Function.fd_seek, Resolver.resolve(wasi_module_name, "fd_seek").?);
    try std.testing.expectEqual(Function.fd_write, Resolver.resolve(wasi_module_name, "fd_write").?);
    try std.testing.expectEqual(Function.path_filestat_get, Resolver.resolve(wasi_module_name, "path_filestat_get").?);
    try std.testing.expectEqual(Function.path_open, Resolver.resolve(wasi_module_name, "path_open").?);
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

test "resolver exposes wasi stdin reads" {
    var memory_bytes = [_]u8{0} ** 128;
    var memory = wasi_nn_abi.LinearMemory.init(&memory_bytes);
    var resolver = Resolver.initWasiConfig(std.testing.allocator, &memory, .{
        .stdin = "robot",
    });
    defer resolver.deinit();

    try memory.writeU32(0, 32);
    try memory.writeU32(4, 3);
    try memory.writeU32(8, 48);
    try memory.writeU32(12, 4);

    try std.testing.expectEqual(
        wasiErrnoSuccess,
        resolver.call(.fd_read, &.{ .{ .i32 = 0 }, .{ .i32 = 0 }, .{ .i32 = 2 }, .{ .i32 = 24 } }),
    );
    try std.testing.expectEqual(@as(u32, 5), try memory.readU32(24));
    try std.testing.expectEqualStrings("rob", try memory.read(32, 3));
    try std.testing.expectEqualStrings("ot", try memory.read(48, 2));
}

test "resolver exposes wasi preopen discovery and readonly file reads" {
    var memory_bytes = [_]u8{0} ** 512;
    var memory = wasi_nn_abi.LinearMemory.init(&memory_bytes);
    const preopens = [_]Preopen{.{ .name = "." }};
    const files = [_]ReadonlyFile{.{ .preopen_name = ".", .path = "policy/input.txt", .data = "action-data" }};
    var resolver = Resolver.initWasiConfig(std.testing.allocator, &memory, .{
        .preopens = &preopens,
        .readonly_files = &files,
    });
    defer resolver.deinit();

    try std.testing.expectEqual(
        wasiErrnoSuccess,
        resolver.call(.fd_prestat_get, &.{ .{ .i32 = 3 }, .{ .i32 = 0 } }),
    );
    try std.testing.expectEqual(@as(u8, 0), (try memory.read(0, 1))[0]);
    try std.testing.expectEqual(@as(u32, 1), try memory.readU32(4));

    try std.testing.expectEqual(
        wasiErrnoSuccess,
        resolver.call(.fd_prestat_dir_name, &.{ .{ .i32 = 3 }, .{ .i32 = 16 }, .{ .i32 = 1 } }),
    );
    try std.testing.expectEqualStrings(".", try memory.read(16, 1));

    try memory.write(32, "policy/input.txt");
    try memory.write(180, "policy");
    try std.testing.expectEqual(
        wasiErrnoSuccess,
        resolver.call(.path_filestat_get, &.{ .{ .i32 = 3 }, .{ .i32 = 0 }, .{ .i32 = 180 }, .{ .i32 = 6 }, .{ .i32 = 208 } }),
    );
    try std.testing.expectEqual(@as(u8, wasiFiletypeDirectory), (try memory.read(224, 1))[0]);

    try std.testing.expectEqual(
        wasiErrnoSuccess,
        resolver.call(.fd_readdir, &.{ .{ .i32 = 3 }, .{ .i32 = 256 }, .{ .i32 = 96 }, .{ .i64 = 0 }, .{ .i32 = 248 } }),
    );
    try std.testing.expectEqual(@as(u32, 30), try memory.readU32(248));
    try std.testing.expectEqual(@as(u32, 6), try memory.readU32(272));
    try std.testing.expectEqual(@as(u8, wasiFiletypeDirectory), (try memory.read(276, 1))[0]);
    try std.testing.expectEqualStrings("policy", try memory.read(280, 6));

    try std.testing.expectEqual(
        wasiErrnoSuccess,
        resolver.call(.path_filestat_get, &.{ .{ .i32 = 3 }, .{ .i32 = 0 }, .{ .i32 = 32 }, .{ .i32 = 16 }, .{ .i32 = 64 } }),
    );
    try std.testing.expectEqual(@as(u8, wasiFiletypeRegularFile), (try memory.read(80, 1))[0]);
    try std.testing.expectEqual(@as(u64, 11), readU64Little(try memory.read(96, 8)));

    try std.testing.expectEqual(
        wasiErrnoSuccess,
        resolver.call(.path_open, &.{
            .{ .i32 = 3 },
            .{ .i32 = 0 },
            .{ .i32 = 32 },
            .{ .i32 = 16 },
            .{ .i32 = 0 },
            .{ .i64 = 0 },
            .{ .i64 = 0 },
            .{ .i32 = 0 },
            .{ .i32 = 128 },
        }),
    );
    const fd = try memory.readU32(128);
    try std.testing.expectEqual(@as(u32, 4), fd);

    try memory.writeU32(136, 160);
    try memory.writeU32(140, 6);
    try std.testing.expectEqual(
        wasiErrnoSuccess,
        resolver.call(.fd_read, &.{ .{ .i32 = fd }, .{ .i32 = 136 }, .{ .i32 = 1 }, .{ .i32 = 152 } }),
    );
    try std.testing.expectEqual(@as(u32, 6), try memory.readU32(152));
    try std.testing.expectEqualStrings("action", try memory.read(160, 6));

    try std.testing.expectEqual(
        wasiErrnoSuccess,
        resolver.call(.fd_seek, &.{ .{ .i32 = fd }, .{ .i64 = 0 }, .{ .i32 = 0 }, .{ .i32 = 176 } }),
    );
    try std.testing.expectEqual(@as(u64, 0), readU64Little(try memory.read(176, 8)));

    try std.testing.expectEqual(wasiErrnoSuccess, resolver.call(.fd_close, &.{.{ .i32 = fd }}));
    try std.testing.expectEqual(
        wasiErrnoBadf,
        resolver.call(.fd_read, &.{ .{ .i32 = fd }, .{ .i32 = 136 }, .{ .i32 = 1 }, .{ .i32 = 152 } }),
    );
}

test "resolver enforces preopen readonly policy" {
    var memory_bytes = [_]u8{0} ** 256;
    var memory = wasi_nn_abi.LinearMemory.init(&memory_bytes);
    const preopens = [_]Preopen{.{ .name = ".", .allow_readdir = false, .allow_path_open = false }};
    const files = [_]ReadonlyFile{.{ .preopen_name = ".", .path = "input.txt", .data = "data" }};
    var resolver = Resolver.initWasiConfig(std.testing.allocator, &memory, .{
        .preopens = &preopens,
        .readonly_files = &files,
    });
    defer resolver.deinit();

    try memory.write(0, "input.txt");
    try std.testing.expectEqual(
        wasiErrnoAcces,
        resolver.call(.path_open, &.{
            .{ .i32 = 3 },
            .{ .i32 = 0 },
            .{ .i32 = 0 },
            .{ .i32 = 9 },
            .{ .i32 = 0 },
            .{ .i64 = 0 },
            .{ .i64 = 0 },
            .{ .i32 = 0 },
            .{ .i32 = 32 },
        }),
    );

    try std.testing.expectEqual(
        wasiErrnoAcces,
        resolver.call(.fd_readdir, &.{ .{ .i32 = 3 }, .{ .i32 = 64 }, .{ .i32 = 64 }, .{ .i64 = 0 }, .{ .i32 = 48 } }),
    );
}
