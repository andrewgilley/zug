const std = @import("std");

const imports = @import("imports.zig");
const module = @import("module.zig");
const memory = @import("memory.zig");

pub const FunctionAddr = enum(u32) { _ };
pub const MemoryAddr = enum(u32) { _ };
pub const TableAddr = enum(u32) { _ };
pub const GlobalAddr = enum(u32) { _ };

pub const FunctionInstance = union(enum) {
    wasm: struct {
        module_instance: u32,
        function: module.Function,
    },
    host: HostFunction,
};

pub const HostFunction = struct {
    function_type: module.FunctionType,
    context: ?*anyopaque,
    callback: *const fn (
        context: ?*anyopaque,
        args: []const module.ConstValue,
        results: []module.ConstValue,
    ) anyerror!void,
};

pub const MemoryInstance = struct {
    bytes: []u8,
    limits: module.Limits,
};

pub const TableInstance = struct {
    elements: []?FunctionAddr,
    limits: module.Limits,
};

pub const GlobalInstance = struct {
    global_type: module.GlobalType,
    value: module.ConstValue,
};

pub const Store = struct {
    allocator: std.mem.Allocator,
    functions: std.ArrayList(FunctionInstance) = .empty,
    memories: std.ArrayList(MemoryInstance) = .empty,
    tables: std.ArrayList(TableInstance) = .empty,
    globals: std.ArrayList(GlobalInstance) = .empty,

    pub fn init(allocator: std.mem.Allocator) Store {
        return .{ .allocator = allocator };
    }

    pub fn deinit(self: *Store) void {

    }
};
