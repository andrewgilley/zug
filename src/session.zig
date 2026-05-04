const std = @import("std");
const tensor = @import("tensor.zig");

pub const Backend = enum {
    onnx,
    wasm,
};

pub const InputSpec = struct {
    name: []const u8,
    path: []const u8,
};

pub const Output = struct {
    name: []const u8,
    value: *const tensor.Tensor,
};

pub const VTable = struct {
    deinit: *const fn (ctx: *anyopaque) void,
    backend: *const fn (ctx: *const anyopaque) Backend,
    load_inputs_from_files: *const fn (ctx: *anyopaque, inputs: []const InputSpec) anyerror!void,
    execute: *const fn (ctx: *anyopaque) anyerror![]const Output,
};

pub const Session = struct {
    ctx: *anyopaque,
    vtable: *const VTable,

    pub fn deinit(self: *Session) void {
        self.vtable.deinit(self.ctx);
        self.* = undefined;
    }

    pub fn backend(self: Session) Backend {
        return self.vtable.backend(self.ctx);
    }

    pub fn loadInputsFromFiles(self: *Session, inputs: []const InputSpec) !void {
        try self.vtable.load_inputs_from_files(self.ctx, inputs);
    }

    pub fn execute(self: *Session) ![]const Output {
        return try self.vtable.execute(self.ctx);
    }
};
