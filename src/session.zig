const std = @import("std");
const onnx = @import("proto/onnx.pb.zig");
const executor = @import("executor.zig");
const tensor = @import("tensor.zig");

pub const max_model_bytes = 100 * 1024 * 1024;

pub const Backend = enum {
    onnx,
    wasm,
};

pub const InputSpec = executor.InputSpec;
pub const Output = executor.Output;

pub const VTable = struct {
    deinit: *const fn (ctx: *anyopaque) void,
    backend: *const fn (ctx: *const anyopaque) Backend,
    set_input: *const fn (ctx: *anyopaque, name: []const u8, value: tensor.Tensor) anyerror!void,
    set_input_by_index: *const fn (ctx: *anyopaque, index: usize, value: tensor.Tensor) anyerror!void,
    load_inputs_from_files: *const fn (ctx: *anyopaque, inputs: []const InputSpec) anyerror!void,
    execute: *const fn (ctx: *anyopaque) anyerror![]const Output,
    output_count: *const fn (ctx: *const anyopaque) anyerror!usize,
    output: *const fn (ctx: *const anyopaque, index: usize) anyerror!Output,
    output_by_name: *const fn (ctx: *const anyopaque, name: []const u8) anyerror!Output,
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

    pub fn setInput(self: *Session, name: []const u8, value: tensor.Tensor) !void {
        try self.vtable.set_input(self.ctx, name, value);
    }

    pub fn setInputByIndex(self: *Session, index: usize, value: tensor.Tensor) !void {
        try self.vtable.set_input_by_index(self.ctx, index, value);
    }

    pub fn setInputCopy(
        self: *Session,
        allocator: std.mem.Allocator,
        name: []const u8,
        value: *const tensor.Tensor,
    ) !void {
        var owned = try cloneTensor(allocator, value);
        errdefer owned.deinit(allocator);

        try self.setInput(name, owned);
    }

    pub fn setInputCopyByIndex(
        self: *Session,
        allocator: std.mem.Allocator,
        index: usize,
        value: *const tensor.Tensor,
    ) !void {
        var owned = try cloneTensor(allocator, value);
        errdefer owned.deinit(allocator);

        try self.setInputByIndex(index, owned);
    }

    pub fn loadInputsFromFiles(self: *Session, inputs: []const InputSpec) !void {
        try self.vtable.load_inputs_from_files(self.ctx, inputs);
    }

    pub fn execute(self: *Session) ![]const Output {
        return try self.vtable.execute(self.ctx);
    }

    pub fn outputCount(self: Session) !usize {
        return try self.vtable.output_count(self.ctx);
    }

    pub fn output(self: Session, index: usize) !Output {
        return try self.vtable.output(self.ctx, index);
    }

    pub fn outputByName(self: Session, name: []const u8) !Output {
        return try self.vtable.output_by_name(self.ctx, name);
    }
};

pub fn loadOnnxFromFile(
    allocator: std.mem.Allocator,
    path: []const u8,
) !Session {
    const bytes = try std.Io.Dir.cwd().readFileAlloc(
        std.Options.debug_io,
        path,
        allocator,
        .limited(max_model_bytes),
    );
    defer allocator.free(bytes);

    return try loadOnnxFromBytes(allocator, bytes);
}

pub fn loadOnnxFromBytes(
    allocator: std.mem.Allocator,
    bytes: []const u8,
) !Session {
    if (bytes.len > max_model_bytes) return error.ModelTooLarge;

    var reader: std.Io.Reader = .fixed(bytes);
    const model = try onnx.ModelProto.decode(&reader, allocator);

    return try initOnnx(allocator, model);
}

pub fn initOnnx(
    allocator: std.mem.Allocator,
    model: onnx.ModelProto,
) !Session {
    const ctx = allocator.create(OnnxSession) catch |err| {
        var owned_model = model;
        owned_model.deinit(allocator);
        return err;
    };

    ctx.* = .{
        .allocator = allocator,
        .model = model,
        .run = undefined,
    };

    ctx.run = executor.Executor.init(allocator, &ctx.model) catch |err| {
        ctx.model.deinit(allocator);
        allocator.destroy(ctx);
        return err;
    };

    return .{
        .ctx = ctx,
        .vtable = &onnx_vtable,
    };
}

const OnnxSession = struct {
    allocator: std.mem.Allocator,
    model: onnx.ModelProto,
    run: executor.Executor,
    outputs: []const Output = &.{},
    computed: bool = false,

    fn deinit(ctx: *anyopaque) void {
        const self = asOnnx(ctx);
        const allocator = self.allocator;

        self.run.deinit();
        self.model.deinit(allocator);
        allocator.destroy(self);
    }

    fn backend(ctx: *const anyopaque) Backend {
        _ = ctx;
        return .onnx;
    }

    fn setInput(ctx: *anyopaque, name: []const u8, value: tensor.Tensor) !void {
        const self = asOnnx(ctx);
        try self.run.setInput(name, value);
        self.outputs = &.{};
        self.computed = false;
    }

    fn setInputByIndex(ctx: *anyopaque, index: usize, value: tensor.Tensor) !void {
        const self = asOnnx(ctx);
        const name = try graphInputNameByIndex(&self.model, index);

        try self.run.setInput(name, value);
        self.outputs = &.{};
        self.computed = false;
    }

    fn loadInputsFromFiles(ctx: *anyopaque, inputs: []const InputSpec) !void {
        const self = asOnnx(ctx);
        try self.run.loadInputsFromFiles(inputs);
        self.outputs = &.{};
        self.computed = false;
    }

    fn execute(ctx: *anyopaque) ![]const Output {
        const self = asOnnx(ctx);
        self.outputs = try self.run.execute();
        self.computed = true;
        return self.outputs;
    }

    fn outputCount(ctx: *const anyopaque) !usize {
        const self = asConstOnnx(ctx);
        if (!self.computed) return error.SessionNotExecuted;

        return self.outputs.len;
    }

    fn output(ctx: *const anyopaque, index: usize) !Output {
        const self = asConstOnnx(ctx);
        if (!self.computed) return error.SessionNotExecuted;
        if (index >= self.outputs.len) return error.OutputIndexOutOfBounds;

        return self.outputs[index];
    }

    fn outputByName(ctx: *const anyopaque, name: []const u8) !Output {
        const self = asConstOnnx(ctx);
        if (!self.computed) return error.SessionNotExecuted;

        for (self.outputs) |item| {
            if (std.mem.eql(u8, item.name, name)) return item;
        }

        return error.OutputNameNotFound;
    }
};

const onnx_vtable = VTable{
    .deinit = OnnxSession.deinit,
    .backend = OnnxSession.backend,
    .set_input = OnnxSession.setInput,
    .set_input_by_index = OnnxSession.setInputByIndex,
    .load_inputs_from_files = OnnxSession.loadInputsFromFiles,
    .execute = OnnxSession.execute,
    .output_count = OnnxSession.outputCount,
    .output = OnnxSession.output,
    .output_by_name = OnnxSession.outputByName,
};

fn asOnnx(ctx: *anyopaque) *OnnxSession {
    return @ptrCast(@alignCast(ctx));
}

fn asConstOnnx(ctx: *const anyopaque) *const OnnxSession {
    return @ptrCast(@alignCast(ctx));
}

fn cloneTensor(allocator: std.mem.Allocator, value: *const tensor.Tensor) !tensor.Tensor {
    return switch (value.data) {
        .float32 => |data| tensor.Tensor.initFloat32(allocator, value.shape, data),
        .int64 => |data| tensor.Tensor.initInt64(allocator, value.shape, data),
        .int32 => |data| tensor.Tensor.initInt32(allocator, value.shape, data),
        .uint8 => |data| tensor.Tensor.initUint8(allocator, value.shape, data),
        .bool => |data| tensor.Tensor.initBool(allocator, value.shape, data),
    };
}

fn graphInputNameByIndex(model: *const onnx.ModelProto, target_index: usize) ![]const u8 {
    const graph = model.graph orelse return error.MissingGraph;

    var visible_index: usize = 0;
    for (graph.input.items) |*input| {
        const name = input.name orelse return error.MissingInputName;
        if (isInitializerName(&graph, name)) continue;

        if (visible_index == target_index) return name;
        visible_index += 1;
    }

    return error.InputIndexOutOfBounds;
}

fn isInitializerName(graph: *const onnx.GraphProto, name: []const u8) bool {
    for (graph.initializer.items) |*initializer| {
        if (initializer.name) |initializer_name| {
            if (std.mem.eql(u8, initializer_name, name)) return true;
        }
    }

    return false;
}

test "onnx session executes model with copied input by index" {
    const allocator = std.testing.allocator;

    var session = try loadOnnxFromFile(allocator, "models/tiny_mnist.onnx");
    defer session.deinit();

    try std.testing.expectEqual(Backend.onnx, session.backend());
    try std.testing.expectError(error.SessionNotExecuted, session.outputCount());

    var input = try tensor.Tensor.initZerosFloat32(allocator, &.{ 1, 1, 28, 28 });
    defer input.deinit(allocator);

    try session.setInputCopyByIndex(allocator, 0, &input);

    const outputs = try session.execute();
    try std.testing.expectEqual(@as(usize, 1), outputs.len);
    try std.testing.expectEqual(@as(usize, 1), try session.outputCount());

    const by_index = try session.output(0);
    const by_name = try session.outputByName("probabilities");

    try std.testing.expectEqualStrings("probabilities", by_index.name);
    try std.testing.expectEqual(by_index.value, by_name.value);
    try std.testing.expectEqual(tensor.DType.float32, by_index.value.dtype);
    try std.testing.expectEqualSlices(usize, &.{ 1, 10 }, by_index.value.shape);

    const values = try by_index.value.float32Data();
    for (values) |value| {
        try std.testing.expectApproxEqAbs(@as(f32, 0.1), value, 0.00001);
    }
}
