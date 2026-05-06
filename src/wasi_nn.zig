const std = @import("std");
const accelerator = @import("accelerator.zig");
const session = @import("session.zig");
const tensor = @import("tensor.zig");

const max_model_bytes = session.max_model_bytes;

pub const GraphEncoding = enum {
    onnx,
};

pub const ExecutionTarget = enum {
    cpu,
    gpu,
    tpu,
    cuda,
    tensorrt,
    rocm,
    vulkan,
    metal,
    directml,
    openvino,
    coreml,
    nnapi,
    webgpu,
    edge_tpu,
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
    backend: accelerator.BackendKind,
    model_bytes: []u8,

    fn deinit(self: *Graph, allocator: std.mem.Allocator) void {
        allocator.free(self.model_bytes);
        self.* = undefined;
    }
};

const ExecutionContext = struct {
    graph: *Graph,
    run: session.Session,

    fn deinit(self: *ExecutionContext) void {
        self.run.deinit();
        self.* = undefined;
    }
};

pub const Host = struct {
    allocator: std.mem.Allocator,
    accelerators: accelerator.Capabilities = accelerator.defaultCapabilities(),
    graphs: std.ArrayList(*Graph) = .empty,
    contexts: std.ArrayList(*ExecutionContext) = .empty,

    pub fn init(allocator: std.mem.Allocator) Host {
        return .{
            .allocator = allocator,
        };
    }

    pub fn initWithAccelerators(
        allocator: std.mem.Allocator,
        accelerators: accelerator.Capabilities,
    ) Host {
        return .{
            .allocator = allocator,
            .accelerators = accelerators,
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
        const backend = try self.resolveExecutionBackend(target);
        if (model_bytes.len > max_model_bytes) return error.ModelTooLarge;

        var validated = try session.loadOnnxFromBytes(self.allocator, model_bytes);
        defer validated.deinit();

        const owned_model_bytes = try self.allocator.dupe(u8, model_bytes);
        errdefer self.allocator.free(owned_model_bytes);

        const graph = try self.allocator.create(Graph);
        errdefer self.allocator.destroy(graph);

        graph.* = .{
            .encoding = encoding,
            .target = target,
            .backend = backend,
            .model_bytes = owned_model_bytes,
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

        var run = try session.loadOnnxFromBytes(self.allocator, graph.model_bytes);
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

        try context.run.setInputCopy(self.allocator, name, value);
    }

    pub fn setInputByIndex(
        self: *Host,
        context_handle: ExecutionContextHandle,
        index: usize,
        value: *const tensor.Tensor,
    ) !void {
        try requireFloat32(value);

        const context = try self.contextPtr(context_handle);

        try context.run.setInputCopyByIndex(self.allocator, index, value);
    }

    pub fn compute(
        self: *Host,
        context_handle: ExecutionContextHandle,
    ) !void {
        const context = try self.contextPtr(context_handle);

        _ = try context.run.execute();
    }

    pub fn outputCount(
        self: *Host,
        context_handle: ExecutionContextHandle,
    ) !usize {
        const context = try self.contextPtr(context_handle);

        return context.run.outputCount() catch |err| switch (err) {
            error.SessionNotExecuted => error.ContextNotComputed,
            else => err,
        };
    }

    pub fn getOutput(
        self: *Host,
        context_handle: ExecutionContextHandle,
        index: usize,
    ) !Output {
        const context = try self.contextPtr(context_handle);

        const output = context.run.output(index) catch |err| switch (err) {
            error.SessionNotExecuted => return error.ContextNotComputed,
            else => return err,
        };
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

    fn resolveExecutionBackend(self: Host, target: ExecutionTarget) !accelerator.BackendKind {
        const backend = switch (target) {
            .cpu => accelerator.BackendKind.cpu,
            .gpu => self.accelerators.defaultGpuBackend() orelse return error.UnsupportedExecutionTarget,
            .tpu => accelerator.BackendKind.edge_tpu,
            .cuda => accelerator.BackendKind.cuda,
            .tensorrt => accelerator.BackendKind.tensorrt,
            .rocm => accelerator.BackendKind.rocm,
            .vulkan => accelerator.BackendKind.vulkan,
            .metal => accelerator.BackendKind.metal,
            .directml => accelerator.BackendKind.directml,
            .openvino => accelerator.BackendKind.openvino,
            .coreml => accelerator.BackendKind.coreml,
            .nnapi => accelerator.BackendKind.nnapi,
            .webgpu => accelerator.BackendKind.webgpu,
            .edge_tpu => accelerator.BackendKind.edge_tpu,
        };

        if (!self.accelerators.supportsGraphExecution(backend)) {
            return error.UnsupportedExecutionTarget;
        }

        return backend;
    }
};

fn requireFloat32(value: *const tensor.Tensor) !void {
    _ = try value.float32Data();
}

test "host rejects unavailable accelerator graph targets" {
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

    try std.testing.expectError(error.UnsupportedExecutionTarget, host.loadGraph(.onnx, .cuda, model_bytes));
}

test "host accepts mock accelerator graph target through backend catalog" {
    const allocator = std.testing.allocator;

    const model_bytes = try std.Io.Dir.cwd().readFileAlloc(
        std.Options.debug_io,
        "models/tiny_mnist.onnx",
        allocator,
        .limited(max_model_bytes),
    );
    defer allocator.free(model_bytes);

    var host = Host.initWithAccelerators(allocator, accelerator.mockEdgeCapabilities());
    defer host.deinit();

    const graph = try host.loadGraph(.onnx, .gpu, model_bytes);
    try std.testing.expectEqual(@as(usize, 0), graph.index);
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
    try std.testing.expectError(error.ContextNotComputed, host.outputCount(context));

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
