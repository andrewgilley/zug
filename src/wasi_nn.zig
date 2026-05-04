const std = @import("std");
const onnx = @import("proto/onnx.pb.zig");
const executor = @import("executor.zig");
const tensor = @import("tensor.zig");

const max_model_bytes = 100 * 1024 * 1024;

pub const GraphEncoding = enum {
    onnx,
};

pub const ExecutionTarget = enum {
    cpu,
};

pub const GraphHandle = struct {
    index: usize,
};

pub const ExecutionContextHandle = struct {
    index: usize,
};

pub const Output = struct {
    name: []const u8,
    value: *const tensor.Tensor,
};

pub const TensorDescriptor = struct {
    dtype: tensor.DType,
    shape: []const usize,
};

const Graph = struct {
    encoding: GraphEncoding,
    target: ExecutionTarget,
    model: onnx.ModelProto,

    fn deinit(self: *Graph, allocator: std.mem.Allocator) void {
        self.model.deinit(allocator);
        self.* = undefined;
    }
};

const ExecutionContext = struct {
    graph: *Graph,
    run: executor.Executor,
    outputs: []const executor.Output = &.{},
    computed: bool = false,

    fn deinit(self: *ExecutionContext) void {
        self.run.deinit();
        self.* = undefined;
    }
};

pub const Host = struct {
    allocator: std.mem.Allocator,
    graphs: std.ArrayList(*Graph) = .empty,
    contexts: std.ArrayList(*ExecutionContext) = .empty,

    pub fn init(allocator: std.mem.Allocator) Host {
        return .{
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Host) void {
        for (self.contexts.items) |context| {
            context.deinit();
            self.allocator.destroy(context);
        }
        self.contexts.deinit(self.allocator);

        for (self.graphs.items) |graph| {
            graph.deinit(self.allocator);
            self.allocator.destroy(graph);
        }
        self.graphs.deinit(self.allocator);

        self.* = undefined;
    }

    pub fn loadGraph(
        self: *Host,
        encoding: GraphEncoding,
        target: ExecutionTarget,
        model_bytes: []const u8,
    ) !GraphHandle {
        if (encoding != .onnx) return error.UnsupportedGraphEncoding;
        if (target != .cpu) return error.UnsupportedExecutionTarget;
        if (model_bytes.len > max_model_bytes) return error.ModelTooLarge;

        var reader: std.Io.Reader = .fixed(model_bytes);

        const graph = try self.allocator.create(Graph);
        errdefer self.allocator.destroy(graph);

        graph.* = .{
            .encoding = encoding,
            .target = target,
            .model = try onnx.ModelProto.decode(&reader, self.allocator),
        };
        errdefer graph.deinit(self.allocator);

        const handle: GraphHandle = .{ .index = self.graphs.items.len };
        try self.graphs.append(self.allocator, graph);

        return handle;
    }

    pub fn initExecutionContext(
        self: *Host,
        graph_handle: GraphHandle,
    ) !ExecutionContextHandle {
        const graph = try self.graphPtr(graph_handle);

        var run = try executor.Executor.init(self.allocator, &graph.model);
        errdefer run.deinit();

        const context = try self.allocator.create(ExecutionContext);
        errdefer self.allocator.destroy(context);

        context.* = .{
            .graph = graph,
            .run = run,
        };

        const handle: ExecutionContextHandle = .{ .index = self.contexts.items.len };
        try self.contexts.append(self.allocator, context);

        return handle;
    }

    pub fn setInput(
        self: *Host,
        context_handle: ExecutionContextHandle,
        name: []const u8,
        value: *const tensor.Tensor,
    ) !void {
        try requireFloat32(value);

        const context = try self.contextPtr(context_handle);
        const owned = try cloneTensor(self.allocator, value);

        try context.run.setInput(name, owned);
        context.computed = false;
        context.outputs = &.{};
    }

    pub fn setInputByIndex(
        self: *Host,
        context_handle: ExecutionContextHandle,
        index: usize,
        value: *const tensor.Tensor,
    ) !void {
        const context = try self.contextPtr(context_handle);
        const name = try graphInputNameByIndex(context.graph, index);

        try self.setInput(context_handle, name, value);
    }

    pub fn compute(
        self: *Host,
        context_handle: ExecutionContextHandle,
    ) !void {
        const context = try self.contextPtr(context_handle);

        context.outputs = try context.run.execute();
        context.computed = true;
    }

    pub fn outputCount(
        self: *Host,
        context_handle: ExecutionContextHandle,
    ) !usize {
        const context = try self.contextPtr(context_handle);
        if (!context.computed) return error.ContextNotComputed;

        return context.outputs.len;
    }

    pub fn getOutput(
        self: *Host,
        context_handle: ExecutionContextHandle,
        index: usize,
    ) !Output {
        const context = try self.contextPtr(context_handle);
        if (!context.computed) return error.ContextNotComputed;
        if (index >= context.outputs.len) return error.OutputIndexOutOfBounds;

        const output = context.outputs[index];
        return .{
            .name = output.name,
            .value = output.value,
        };
    }

    pub fn getOutputDescriptor(
        self: *Host,
        context_handle: ExecutionContextHandle,
        index: usize,
    ) !TensorDescriptor {
        const output = try self.getOutput(context_handle, index);

        return .{
            .dtype = output.value.dtype,
            .shape = output.value.shape,
        };
    }

    fn graphPtr(self: *Host, handle: GraphHandle) !*Graph {
        if (handle.index >= self.graphs.items.len) return error.InvalidGraphHandle;
        return self.graphs.items[handle.index];
    }

    fn contextPtr(self: *Host, handle: ExecutionContextHandle) !*ExecutionContext {
        if (handle.index >= self.contexts.items.len) return error.InvalidExecutionContextHandle;
        return self.contexts.items[handle.index];
    }
};

fn requireFloat32(value: *const tensor.Tensor) !void {
    _ = try value.float32Data();
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

fn graphInputNameByIndex(graph: *const Graph, target_index: usize) ![]const u8 {
    const graph_proto = graph.model.graph orelse return error.MissingGraph;

    var visible_index: usize = 0;
    for (graph_proto.input.items) |*input| {
        const name = input.name orelse return error.MissingInputName;
        if (isInitializerName(&graph_proto, name)) continue;

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

test "host runs onnx graph through wasi-nn shaped calls" {
    const allocator = std.testing.allocator;

    const model_bytes = try std.Io.Dir.cwd().readFileAlloc(
        std.Options.debug_io,
        "models/tiny_mnist.onnx",
        allocator,
        .limited(max_model_bytes),
    );
    defer allocator.free(model_bytes);

    var host = Host.init(allocator);
    defer host.deinit();

    const graph = try host.loadGraph(.onnx, .cpu, model_bytes);
    const context = try host.initExecutionContext(graph);

    const input_shape = [_]usize{ 1, 1, 28, 28 };
    var input = try tensor.Tensor.initZerosFloat32(allocator, &input_shape);
    defer input.deinit(allocator);

    try host.setInputByIndex(context, 0, &input);
    try host.compute(context);

    try std.testing.expectEqual(@as(usize, 1), try host.outputCount(context));

    const output = try host.getOutput(context, 0);
    try std.testing.expectEqualStrings("probabilities", output.name);
    try std.testing.expectEqual(tensor.DType.float32, output.value.dtype);
    try std.testing.expectEqualSlices(usize, &.{ 1, 10 }, output.value.shape);

    const values = try output.value.float32Data();
    for (values) |value| {
        try std.testing.expectApproxEqAbs(@as(f32, 0.1), value, 0.00001);
    }
}
