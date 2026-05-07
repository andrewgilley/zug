const std = @import("std");
const tensor = @import("tensor.zig");
const wasi_nn = @import("wasi_nn.zig");

const max_model_bytes = 100 * 1024 * 1024;

pub const Host = wasi_nn.Host;

pub const Status = enum(u32) {
    ok = 0,
    invalid_memory = 1,
    invalid_encoding = 2,
    invalid_target = 3,
    invalid_dtype = 4,
    unsupported_dtype = 5,
    invalid_graph_handle = 6,
    invalid_context_handle = 7,
    invalid_input = 8,
    invalid_output = 9,
    context_not_computed = 10,
    buffer_too_small = 11,
    model_too_large = 12,
    runtime_error = 13,
    missing_model = 14,
};

pub const GraphEncoding = enum(u32) {
    onnx = 0,
};

pub const ExecutionTarget = enum(u32) {
    cpu = 0,
    gpu = 1,
    tpu = 2,
    cuda = 100,
    tensorrt = 101,
    rocm = 102,
    vulkan = 103,
    metal = 104,
    directml = 105,
    openvino = 106,
    coreml = 107,
    nnapi = 108,
    webgpu = 109,
    edge_tpu = 110,
};

pub const DType = enum(u32) {
    float32 = 1,
    int64 = 2,
    int32 = 3,
    uint8 = 4,
    bool = 5,
};

pub const LinearMemory = struct {
    bytes: []u8,

    pub fn init(bytes: []u8) LinearMemory {
        return .{ .bytes = bytes };
    }

    pub fn read(self: *const LinearMemory, ptr: u32, len: u32) ![]const u8 {
        const range = try checkedRange(self.bytes.len, ptr, len);
        return self.bytes[range.start..range.end];
    }

    pub fn write(self: *LinearMemory, ptr: u32, data: []const u8) !void {
        const len = std.math.cast(u32, data.len) orelse return error.InvalidMemoryRange;
        const range = try checkedRange(self.bytes.len, ptr, len);
        @memcpy(self.bytes[range.start..range.end], data);
    }

    pub fn readU32(self: *const LinearMemory, ptr: u32) !u32 {
        const bytes = try self.read(ptr, 4);
        return readU32Little(bytes);
    }

    pub fn writeU32(self: *LinearMemory, ptr: u32, value: u32) !void {
        const bytes = try self.writeSlice(ptr, 4);
        writeU32Little(bytes, value);
    }

    pub fn writeU64(self: *LinearMemory, ptr: u32, value: u64) !void {
        const bytes = try self.writeSlice(ptr, 8);
        writeU64Little(bytes, value);
    }

    pub fn writeSlice(self: *LinearMemory, ptr: u32, len: u32) ![]u8 {
        const range = try checkedRange(self.bytes.len, ptr, len);
        return self.bytes[range.start..range.end];
    }
};

pub const Surface = struct {
    allocator: std.mem.Allocator,
    host: *wasi_nn.Host,
    memory: *LinearMemory,
    preloaded_model: ?[]const u8 = null,

    pub fn init(
        allocator: std.mem.Allocator,
        host: *wasi_nn.Host,
        memory: *LinearMemory,
    ) Surface {
        return .{
            .allocator = allocator,
            .host = host,
            .memory = memory,
        };
    }

    pub fn setPreloadedModel(self: *Surface, model_bytes: ?[]const u8) void {
        self.preloaded_model = model_bytes;
    }

    pub fn loadGraph(
        self: *Surface,
        model_ptr: u32,
        model_len: u32,
        encoding_code: u32,
        target_code: u32,
        out_graph_handle_ptr: u32,
    ) Status {
        const model_bytes = self.memory.read(model_ptr, model_len) catch return .invalid_memory;

        return self.loadGraphBytes(model_bytes, encoding_code, target_code, out_graph_handle_ptr);
    }

    pub fn loadPreloadedGraph(
        self: *Surface,
        encoding_code: u32,
        target_code: u32,
        out_graph_handle_ptr: u32,
    ) Status {
        const model_bytes = self.preloaded_model orelse return .missing_model;

        return self.loadGraphBytes(model_bytes, encoding_code, target_code, out_graph_handle_ptr);
    }

    fn loadGraphBytes(
        self: *Surface,
        model_bytes: []const u8,
        encoding_code: u32,
        target_code: u32,
        out_graph_handle_ptr: u32,
    ) Status {
        const encoding = decodeGraphEncoding(encoding_code) catch return .invalid_encoding;
        const target = decodeExecutionTarget(target_code) catch return .invalid_target;

        const handle = self.host.loadGraph(encoding, target, model_bytes) catch |err| {
            return statusFromError(err);
        };

        const index = std.math.cast(u32, handle.index) orelse return .runtime_error;
        self.memory.writeU32(out_graph_handle_ptr, index) catch return .invalid_memory;

        return .ok;
    }

    pub fn initExecutionContext(
        self: *Surface,
        graph_handle: u32,
        out_context_handle_ptr: u32,
    ) Status {
        const handle = self.host.initExecutionContext(.{ .index = graph_handle }) catch |err| {
            return statusFromError(err);
        };

        const index = std.math.cast(u32, handle.index) orelse return .runtime_error;
        self.memory.writeU32(out_context_handle_ptr, index) catch return .invalid_memory;

        return .ok;
    }

    pub fn setInputByIndex(
        self: *Surface,
        context_handle: u32,
        input_index: u32,
        dtype_code: u32,
        shape_ptr: u32,
        shape_len: u32,
        data_ptr: u32,
        data_len: u32,
    ) Status {
        const dtype = decodeDType(dtype_code) catch return .invalid_dtype;
        if (dtype != .float32) return .unsupported_dtype;

        const shape = readShape(self.allocator, self.memory, shape_ptr, shape_len) catch |err| {
            return statusFromError(err);
        };
        defer self.allocator.free(shape);

        const data_bytes = self.memory.read(data_ptr, data_len) catch return .invalid_memory;

        var value = tensorFromFloat32Bytes(self.allocator, shape, data_bytes) catch |err| {
            return statusFromError(err);
        };
        defer value.deinit(self.allocator);

        self.host.setInputByIndex(
            .{ .index = context_handle },
            input_index,
            &value,
        ) catch |err| {
            return statusFromError(err);
        };

        return .ok;
    }

    pub fn compute(self: *Surface, context_handle: u32) Status {
        self.host.compute(.{ .index = context_handle }) catch |err| {
            return statusFromError(err);
        };

        return .ok;
    }

    pub fn getOutputDescriptor(
        self: *Surface,
        context_handle: u32,
        output_index: u32,
        out_dtype_ptr: u32,
        out_shape_ptr: u32,
        shape_capacity: u32,
        out_shape_len_ptr: u32,
        out_byte_len_ptr: u32,
    ) Status {
        const descriptor = self.host.getOutputDescriptor(
            .{ .index = context_handle },
            output_index,
        ) catch |err| {
            return statusFromError(err);
        };

        const dtype_code = encodeDType(descriptor.dtype) catch return .unsupported_dtype;
        const shape_len = std.math.cast(u32, descriptor.shape.len) orelse return .runtime_error;
        const byte_len = tensorByteLen(descriptor.dtype, descriptor.shape) catch |err| {
            return statusFromError(err);
        };

        self.memory.writeU32(out_dtype_ptr, @intFromEnum(dtype_code)) catch return .invalid_memory;
        self.memory.writeU32(out_shape_len_ptr, shape_len) catch return .invalid_memory;
        self.memory.writeU32(out_byte_len_ptr, byte_len) catch return .invalid_memory;

        if (shape_capacity < shape_len) return .buffer_too_small;

        var shape_offset = out_shape_ptr;
        for (descriptor.shape) |dim| {
            const dim_u64 = std.math.cast(u64, dim) orelse return .runtime_error;
            self.memory.writeU64(shape_offset, dim_u64) catch return .invalid_memory;
            shape_offset = std.math.add(u32, shape_offset, 8) catch return .invalid_memory;
        }

        return .ok;
    }

    pub fn getOutput(
        self: *Surface,
        context_handle: u32,
        output_index: u32,
        out_data_ptr: u32,
        out_data_len: u32,
        out_bytes_written_ptr: u32,
    ) Status {
        const output = self.host.getOutput(.{ .index = context_handle }, output_index) catch |err| {
            return statusFromError(err);
        };

        const required = tensorByteLen(output.value.dtype, output.value.shape) catch |err| {
            return statusFromError(err);
        };

        self.memory.writeU32(out_bytes_written_ptr, required) catch return .invalid_memory;

        if (out_data_len < required) return .buffer_too_small;

        writeTensorBytes(self.memory, out_data_ptr, output.value) catch |err| {
            return statusFromError(err);
        };

        return .ok;
    }
};

const Range = struct {
    start: usize,
    end: usize,
};

fn checkedRange(memory_len: usize, ptr: u32, len: u32) !Range {
    const start = std.math.cast(usize, ptr) orelse return error.InvalidMemoryRange;
    const byte_len = std.math.cast(usize, len) orelse return error.InvalidMemoryRange;
    const end = try std.math.add(usize, start, byte_len);

    if (end > memory_len) return error.InvalidMemoryRange;

    return .{
        .start = start,
        .end = end,
    };
}

fn decodeGraphEncoding(value: u32) !wasi_nn.GraphEncoding {
    return switch (value) {
        @intFromEnum(GraphEncoding.onnx) => .onnx,
        else => error.InvalidGraphEncoding,
    };
}

fn decodeExecutionTarget(value: u32) !wasi_nn.ExecutionTarget {
    return switch (value) {
        @intFromEnum(ExecutionTarget.cpu) => .cpu,
        @intFromEnum(ExecutionTarget.gpu) => .gpu,
        @intFromEnum(ExecutionTarget.tpu) => .tpu,
        @intFromEnum(ExecutionTarget.cuda) => .cuda,
        @intFromEnum(ExecutionTarget.tensorrt) => .tensorrt,
        @intFromEnum(ExecutionTarget.rocm) => .rocm,
        @intFromEnum(ExecutionTarget.vulkan) => .vulkan,
        @intFromEnum(ExecutionTarget.metal) => .metal,
        @intFromEnum(ExecutionTarget.directml) => .directml,
        @intFromEnum(ExecutionTarget.openvino) => .openvino,
        @intFromEnum(ExecutionTarget.coreml) => .coreml,
        @intFromEnum(ExecutionTarget.nnapi) => .nnapi,
        @intFromEnum(ExecutionTarget.webgpu) => .webgpu,
        @intFromEnum(ExecutionTarget.edge_tpu) => .edge_tpu,
        else => error.InvalidExecutionTarget,
    };
}

fn decodeDType(value: u32) !DType {
    return switch (value) {
        @intFromEnum(DType.float32) => .float32,
        @intFromEnum(DType.int64) => .int64,
        @intFromEnum(DType.int32) => .int32,
        @intFromEnum(DType.uint8) => .uint8,
        @intFromEnum(DType.bool) => .bool,
        else => error.InvalidDType,
    };
}

fn encodeDType(value: tensor.DType) !DType {
    return switch (value) {
        .float32 => .float32,
        .int64 => .int64,
        .int32 => .int32,
        .uint8 => .uint8,
        .bool => .bool,
    };
}

fn readShape(
    allocator: std.mem.Allocator,
    memory: *const LinearMemory,
    shape_ptr: u32,
    shape_len: u32,
) ![]usize {
    const dim_count = std.math.cast(usize, shape_len) orelse return error.InvalidTensorShape;
    const shape = try allocator.alloc(usize, dim_count);
    errdefer allocator.free(shape);

    var offset = shape_ptr;
    for (shape) |*dim| {
        const bytes = try memory.read(offset, 8);
        const value = readU64Little(bytes);
        if (value == 0) return error.InvalidTensorShape;

        dim.* = std.math.cast(usize, value) orelse return error.InvalidTensorShape;
        offset = try std.math.add(u32, offset, 8);
    }

    return shape;
}

fn tensorFromFloat32Bytes(
    allocator: std.mem.Allocator,
    shape: []const usize,
    data_bytes: []const u8,
) !tensor.Tensor {
    const count = try tensor.elementCount(shape);
    const expected_len = try std.math.mul(usize, count, @sizeOf(f32));
    if (data_bytes.len != expected_len) return error.InvalidTensorData;

    const owned_shape = try allocator.dupe(usize, shape);
    errdefer allocator.free(owned_shape);

    const data = try allocator.alloc(f32, count);
    errdefer allocator.free(data);

    for (data, 0..) |*value, index| {
        const offset = index * @sizeOf(f32);
        const bits = readU32Little(data_bytes[offset..][0..4]);
        value.* = @bitCast(bits);
    }

    return tensor.Tensor.initOwnedFloat32(allocator, owned_shape, data);
}

fn tensorByteLen(dtype: tensor.DType, shape: []const usize) !u32 {
    const count = try tensor.elementCount(shape);
    const size: usize = switch (dtype) {
        .float32 => @sizeOf(f32),
        .int64 => @sizeOf(i64),
        .int32 => @sizeOf(i32),
        .uint8 => @sizeOf(u8),
        .bool => @sizeOf(u8),
    };
    const byte_len = try std.math.mul(usize, count, size);

    return std.math.cast(u32, byte_len) orelse error.TensorTooLarge;
}

fn writeTensorBytes(memory: *LinearMemory, ptr: u32, value: *const tensor.Tensor) !void {
    switch (value.data) {
        .float32 => |items| {
            var offset = ptr;
            for (items) |item| {
                const bits: u32 = @bitCast(item);
                const bytes = try memory.writeSlice(offset, 4);
                writeU32Little(bytes, bits);
                offset = try std.math.add(u32, offset, 4);
            }
        },
        .int64 => |items| {
            var offset = ptr;
            for (items) |item| {
                const bits: u64 = @bitCast(item);
                const bytes = try memory.writeSlice(offset, 8);
                writeU64Little(bytes, bits);
                offset = try std.math.add(u32, offset, 8);
            }
        },
        .int32 => |items| {
            var offset = ptr;
            for (items) |item| {
                const bits: u32 = @bitCast(item);
                const bytes = try memory.writeSlice(offset, 4);
                writeU32Little(bytes, bits);
                offset = try std.math.add(u32, offset, 4);
            }
        },
        .uint8 => |items| try memory.write(ptr, items),
        .bool => |items| {
            const len = std.math.cast(u32, items.len) orelse return error.TensorTooLarge;
            const bytes = try memory.writeSlice(ptr, len);
            for (items, bytes) |item, *byte| {
                byte.* = if (item) 1 else 0;
            }
        },
    }
}

fn statusFromError(err: anyerror) Status {
    return switch (err) {
        error.InvalidMemoryRange,
        error.Overflow,
        => .invalid_memory,

        error.InvalidGraphEncoding,
        error.UnsupportedGraphEncoding,
        => .invalid_encoding,

        error.InvalidExecutionTarget,
        error.UnsupportedExecutionTarget,
        => .invalid_target,

        error.InvalidDType => .invalid_dtype,
        error.ExpectedFloat32Tensor => .unsupported_dtype,

        error.InvalidGraphHandle => .invalid_graph_handle,
        error.InvalidExecutionContextHandle => .invalid_context_handle,

        error.ContextNotComputed,
        error.SessionNotExecuted,
        => .context_not_computed,
        error.OutputIndexOutOfBounds => .invalid_output,

        error.InputIndexOutOfBounds,
        error.InvalidTensorShape,
        error.InvalidTensorData,
        error.TensorElementCountMismatch,
        error.ZeroDimensionUnsupported,
        error.InputOverridesInitializer,
        error.UnknownGraphInput,
        => .invalid_input,

        error.ModelTooLarge => .model_too_large,
        error.TensorTooLarge => .buffer_too_small,

        else => .runtime_error,
    };
}

fn readU32Little(bytes: []const u8) u32 {
    return @as(u32, bytes[0]) |
        (@as(u32, bytes[1]) << 8) |
        (@as(u32, bytes[2]) << 16) |
        (@as(u32, bytes[3]) << 24);
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

fn writeU32Little(bytes: []u8, value: u32) void {
    bytes[0] = std.math.cast(u8, value & 0xff) orelse unreachable;
    bytes[1] = std.math.cast(u8, (value >> 8) & 0xff) orelse unreachable;
    bytes[2] = std.math.cast(u8, (value >> 16) & 0xff) orelse unreachable;
    bytes[3] = std.math.cast(u8, (value >> 24) & 0xff) orelse unreachable;
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

test "abi drives wasi-nn host through linear memory" {
    const allocator = std.testing.allocator;

    const model_bytes = try std.Io.Dir.cwd().readFileAlloc(
        std.Options.debug_io,
        "models/tiny_mnist.onnx",
        allocator,
        .limited(max_model_bytes),
    );
    defer allocator.free(model_bytes);

    const memory_bytes = try allocator.alloc(u8, 2 * 1024 * 1024);
    defer allocator.free(memory_bytes);
    @memset(memory_bytes, 0);

    var memory = LinearMemory.init(memory_bytes);
    var host = wasi_nn.Host.init(allocator);
    defer host.deinit();

    var surface = Surface.init(allocator, &host, &memory);

    const graph_handle_ptr: u32 = 16;
    const context_handle_ptr: u32 = 20;
    const dtype_ptr: u32 = 24;
    const shape_len_ptr: u32 = 28;
    const byte_len_ptr: u32 = 32;
    const bytes_written_ptr: u32 = 36;
    const model_ptr: u32 = 1024;
    const shape_ptr: u32 = 200_000;
    const input_ptr: u32 = 201_000;
    const output_shape_ptr: u32 = 300_000;
    const output_ptr: u32 = 301_000;

    try memory.write(model_ptr, model_bytes);

    try std.testing.expectEqual(
        Status.ok,
        surface.loadGraph(
            model_ptr,
            std.math.cast(u32, model_bytes.len) orelse return error.ModelTooLarge,
            @intFromEnum(GraphEncoding.onnx),
            @intFromEnum(ExecutionTarget.cpu),
            graph_handle_ptr,
        ),
    );

    const graph_handle = try memory.readU32(graph_handle_ptr);

    try std.testing.expectEqual(
        Status.ok,
        surface.initExecutionContext(graph_handle, context_handle_ptr),
    );

    const context_handle = try memory.readU32(context_handle_ptr);

    try memory.writeU64(shape_ptr, 1);
    try memory.writeU64(shape_ptr + 8, 1);
    try memory.writeU64(shape_ptr + 16, 28);
    try memory.writeU64(shape_ptr + 24, 28);

    try std.testing.expectEqual(
        Status.ok,
        surface.setInputByIndex(
            context_handle,
            0,
            @intFromEnum(DType.float32),
            shape_ptr,
            4,
            input_ptr,
            1 * 1 * 28 * 28 * @sizeOf(f32),
        ),
    );

    try std.testing.expectEqual(Status.ok, surface.compute(context_handle));

    try std.testing.expectEqual(
        Status.ok,
        surface.getOutputDescriptor(
            context_handle,
            0,
            dtype_ptr,
            output_shape_ptr,
            4,
            shape_len_ptr,
            byte_len_ptr,
        ),
    );

    try std.testing.expectEqual(@intFromEnum(DType.float32), try memory.readU32(dtype_ptr));
    try std.testing.expectEqual(@as(u32, 2), try memory.readU32(shape_len_ptr));
    try std.testing.expectEqual(@as(u32, 40), try memory.readU32(byte_len_ptr));
    try std.testing.expectEqual(@as(u64, 1), readU64Little(try memory.read(output_shape_ptr, 8)));
    try std.testing.expectEqual(@as(u64, 10), readU64Little(try memory.read(output_shape_ptr + 8, 8)));

    try std.testing.expectEqual(
        Status.ok,
        surface.getOutput(context_handle, 0, output_ptr, 40, bytes_written_ptr),
    );
    try std.testing.expectEqual(@as(u32, 40), try memory.readU32(bytes_written_ptr));

    const output_bytes = try memory.read(output_ptr, 40);
    for (0..10) |index| {
        const offset = index * @sizeOf(f32);
        const value: f32 = @bitCast(readU32Little(output_bytes[offset..][0..4]));
        try std.testing.expectApproxEqAbs(@as(f32, 0.1), value, 0.00001);
    }
}

test "abi loads a host preloaded model without guest model bytes" {
    const allocator = std.testing.allocator;

    const model_bytes = try std.Io.Dir.cwd().readFileAlloc(
        std.Options.debug_io,
        "models/tiny_mnist.onnx",
        allocator,
        .limited(max_model_bytes),
    );
    defer allocator.free(model_bytes);

    var memory_bytes = [_]u8{0} ** 64;
    var memory = LinearMemory.init(&memory_bytes);
    var host = wasi_nn.Host.init(allocator);
    defer host.deinit();

    var surface = Surface.init(allocator, &host, &memory);
    const graph_handle_ptr: u32 = 16;

    try std.testing.expectEqual(
        Status.missing_model,
        surface.loadPreloadedGraph(
            @intFromEnum(GraphEncoding.onnx),
            @intFromEnum(ExecutionTarget.cpu),
            graph_handle_ptr,
        ),
    );

    surface.setPreloadedModel(model_bytes);

    try std.testing.expectEqual(
        Status.ok,
        surface.loadPreloadedGraph(
            @intFromEnum(GraphEncoding.onnx),
            @intFromEnum(ExecutionTarget.cpu),
            graph_handle_ptr,
        ),
    );
    try std.testing.expectEqual(@as(u32, 0), try memory.readU32(graph_handle_ptr));
}
