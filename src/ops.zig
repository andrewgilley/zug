const std = @import("std");
const onnx = @import("proto/onnx.pb.zig");
const tensor = @import("tensor.zig");

pub const GemmOptions = struct {
    alpha: f32 = 1.0,
    beta: f32 = 1.0,
    trans_a: bool = false,
    trans_b: bool = false,
};

pub const ConvOptions = struct {
    pads: [4]usize = .{ 0, 0, 0, 0 },
    strides: [2]usize = .{ 1, 1 },
    dilations: [2]usize = .{ 1, 1 },
    group: usize = 1,
};

pub fn constant(allocator: std.mem.Allocator, node: *const onnx.NodeProto) !tensor.Tensor {
    if (node.input.items.len != 0) return error.ConstantUnexpectedInputs;

    for (node.attribute.items) |*attribute| {
        const name = attribute.name orelse continue;

        if (std.mem.eql(u8, name, "value")) {
            const value = attribute.t orelse return error.InvalidConstantValue;
            return tensorFromTensorProto(allocator, &value);
        }

        if (std.mem.eql(u8, name, "value_float")) {
            const value = attribute.f orelse return error.InvalidConstantValue;
            const shape = [_]usize{};
            const data = [_]f32{value};
            return tensor.Tensor.init(allocator, &shape, &data);
        }

        if (std.mem.eql(u8, name, "value_floats")) {
            const shape = [_]usize{attribute.floats.items.len};
            return tensor.Tensor.init(allocator, &shape, attribute.floats.items);
        }

        if (std.mem.startsWith(u8, name, "value_") or
            std.mem.eql(u8, name, "sparse_value"))
        {
            return error.UnsupportedConstantValue;
        }
    }

    return error.MissingConstantValue;
}

pub fn add(
    allocator: std.mem.Allocator,
    a: *const tensor.Tensor,
    b: *const tensor.Tensor,
) !tensor.Tensor {
    const a_data = try a.float32Data();
    const b_data = try b.float32Data();

    if (sameShape(a.shape, b.shape)) {
        var output = try tensor.Tensor.initZeros(allocator, a.shape);
        errdefer output.deinit(allocator);
        const output_data = try output.float32DataMut();

        for (output_data, 0..) |*value, index| {
            value.* = a_data[index] + b_data[index];
        }

        return output;
    }

    if (isScalar(a)) {
        var output = try tensor.Tensor.initZeros(allocator, b.shape);
        errdefer output.deinit(allocator);
        const output_data = try output.float32DataMut();

        for (output_data, 0..) |*value, index| {
            value.* = a_data[0] + b_data[index];
        }

        return output;
    }

    if (isScalar(b)) {
        var output = try tensor.Tensor.initZeros(allocator, a.shape);
        errdefer output.deinit(allocator);
        const output_data = try output.float32DataMut();

        for (output_data, 0..) |*value, index| {
            value.* = a_data[index] + b_data[0];
        }

        return output;
    }

    if (canBroadcastLastDim(a.shape, b.shape)) {
        return addLastDimBroadcast(allocator, a, b);
    }

    if (canBroadcastLastDim(b.shape, a.shape)) {
        return addLastDimBroadcast(allocator, b, a);
    }

    return error.AddShapeMismatch;
}

pub fn conv(
    allocator: std.mem.Allocator,
    x: *const tensor.Tensor,
    w: *const tensor.Tensor,
    b: ?*const tensor.Tensor,
    options: ConvOptions,
) !tensor.Tensor {
    const x_data = try x.float32Data();
    const w_data = try w.float32Data();
    const bias_data = if (b) |bias| try bias.float32Data() else null;

    if (x.shape.len != 4 or w.shape.len != 4) return error.ConvRequiresNchwTensors;
    if (options.group != 1) return error.ConvGroupsUnsupported;
    if (options.dilations[0] != 1 or options.dilations[1] != 1) return error.ConvDilationUnsupported;

    const batch = x.shape[0];
    const in_channels = x.shape[1];
    const input_h = x.shape[2];
    const input_w = x.shape[3];

    const out_channels = w.shape[0];
    const kernel_in_channels = w.shape[1];
    const kernel_h = w.shape[2];
    const kernel_w = w.shape[3];

    if (kernel_in_channels != in_channels) return error.ConvChannelMismatch;
    if (kernel_h == 0 or kernel_w == 0) return error.ConvInvalidKernel;

    if (b) |bias| {
        if (!(bias.shape.len == 1 and bias.shape[0] == out_channels)) {
            return error.ConvBiasShapeMismatch;
        }
    }

    const pad_top = options.pads[0];
    const pad_left = options.pads[1];
    const pad_bottom = options.pads[2];
    const pad_right = options.pads[3];
    const stride_h = options.strides[0];
    const stride_w = options.strides[1];

    if (stride_h == 0 or stride_w == 0) return error.ConvInvalidStride;

    const padded_h = try std.math.add(usize, try std.math.add(usize, input_h, pad_top), pad_bottom);
    const padded_w = try std.math.add(usize, try std.math.add(usize, input_w, pad_left), pad_right);

    if (padded_h < kernel_h or padded_w < kernel_w) return error.ConvKernelLargerThanInput;

    const output_h = ((padded_h - kernel_h) / stride_h) + 1;
    const output_w = ((padded_w - kernel_w) / stride_w) + 1;

    const output_shape = [_]usize{ batch, out_channels, output_h, output_w };
    var output = try tensor.Tensor.initZeros(allocator, &output_shape);
    errdefer output.deinit(allocator);
    const output_data = try output.float32DataMut();

    for (0..batch) |n| {
        for (0..out_channels) |oc| {
            for (0..output_h) |oh| {
                for (0..output_w) |ow| {
                    var sum: f32 = if (bias_data) |bias| bias[oc] else 0.0;

                    for (0..in_channels) |ic| {
                        for (0..kernel_h) |kh| {
                            const padded_y = oh * stride_h + kh;
                            if (padded_y < pad_top) continue;

                            const input_y = padded_y - pad_top;
                            if (input_y >= input_h) continue;

                            for (0..kernel_w) |kw| {
                                const padded_x = ow * stride_w + kw;
                                if (padded_x < pad_left) continue;

                                const input_x = padded_x - pad_left;
                                if (input_x >= input_w) continue;

                                sum += x_data[nchwIndex(x.shape, n, ic, input_y, input_x)] *
                                    w_data[nchwIndex(w.shape, oc, ic, kh, kw)];
                            }
                        }
                    }

                    output_data[nchwIndex(output.shape, n, oc, oh, ow)] = sum;
                }
            }
        }
    }

    return output;
}

pub fn flatten(allocator: std.mem.Allocator, input: *const tensor.Tensor, axis: i64) !tensor.Tensor {
    const input_data = try input.float32Data();
    const normalized_axis = try normalizeAxis(axis, input.shape.len, true);
    const outer = try tensor.elementCount(input.shape[0..normalized_axis]);
    const inner = try tensor.elementCount(input.shape[normalized_axis..]);

    const shape = [_]usize{ outer, inner };

    return tensor.Tensor.init(allocator, &shape, input_data);
}

pub fn gemm(
    allocator: std.mem.Allocator,
    a: *const tensor.Tensor,
    b: *const tensor.Tensor,
    c: ?*const tensor.Tensor,
    options: GemmOptions,
) !tensor.Tensor {
    const a_data = try a.float32Data();
    const b_data = try b.float32Data();

    if (a.shape.len != 2 or b.shape.len != 2) return error.GemmRequiresMatrices;

    const a_rows = if (options.trans_a) a.shape[1] else a.shape[0];
    const a_cols = if (options.trans_a) a.shape[0] else a.shape[1];
    const b_rows = if (options.trans_b) b.shape[1] else b.shape[0];
    const b_cols = if (options.trans_b) b.shape[0] else b.shape[1];

    if (a_cols != b_rows) return error.GemmDimensionMismatch;

    const output_shape = [_]usize{ a_rows, b_cols };
    var output = try tensor.Tensor.initZeros(allocator, &output_shape);
    errdefer output.deinit(allocator);
    const output_data = try output.float32DataMut();

    for (0..a_rows) |row| {
        for (0..b_cols) |col| {
            var sum: f32 = 0.0;

            for (0..a_cols) |k| {
                sum += matrixValue(a_data, a.shape, row, k, options.trans_a) *
                    matrixValue(b_data, b.shape, k, col, options.trans_b);
            }

            const index = row * b_cols + col;
            output_data[index] = options.alpha * sum;

            if (c) |bias| {
                output_data[index] += options.beta * try biasValue(bias, row, col, a_rows, b_cols);
            }
        }
    }

    return output;
}

pub fn matmul(
    allocator: std.mem.Allocator,
    a: *const tensor.Tensor,
    b: *const tensor.Tensor,
) !tensor.Tensor {
    const a_data = try a.float32Data();
    const b_data = try b.float32Data();

    if (a.shape.len != 2 or b.shape.len != 2)  return error.MatMulRequiresMatrices;

    const rows = a.shape[0];
    const shared = a.shape[1];
    const cols = b.shape[1];

    if (shared != b.shape[0]) return error.MatMulDimensionMismatch;

    const output_shape = [_]usize{ rows, cols };
    var output = try tensor.Tensor.initZeros(allocator, &output_shape);
    errdefer output.deinit(allocator);
    const output_data = try output.float32DataMut();

    for (0..rows) |row| {
        for (0..cols) |col| {
            var sum: f32 = 0.0;

            for (0..shared) |k| {
                sum += a_data[row * shared + k] * b_data[k * cols + col];
            }

            output_data[row * cols + col] = sum;
        }
    }

    return output;
}

pub fn softmax(allocator: std.mem.Allocator, input: *const tensor.Tensor, axis: i64) !tensor.Tensor {
    const input_data = try input.float32Data();

    if (input.shape.len == 0) return error.SoftmaxRequiresRankedTensor;

    const normalized_axis = try normalizeAxis(axis, input.shape.len, false);
    const outer = try tensor.elementCount(input.shape[0..normalized_axis]);
    const axis_len = input.shape[normalized_axis];
    const inner = try tensor.elementCount(input.shape[normalized_axis + 1 ..]);

    var output = try tensor.Tensor.initZeros(allocator, input.shape);
    errdefer output.deinit(allocator);
    const output_data = try output.float32DataMut();

    for (0..outer) |outer_index| {
        for (0..inner) |inner_index| {
            const base = outer_index * axis_len * inner + inner_index;
            var max_value = input_data[base];

            for (1..axis_len) |axis_index| {
                const value = input_data[base + axis_index * inner];
                if (value > max_value) max_value = value;
            }

            var sum: f32 = 0.0;

            for (0..axis_len) |axis_index| {
                const index = base + axis_index * inner;
                const value = @exp(input_data[index] - max_value);
                output_data[index] = value;
                sum += value;
            }

            for (0..axis_len) |axis_index| {
                const index = base + axis_index * inner;
                output_data[index] /= sum;
            }
        }
    }

    return output;
}

fn sameShape(a: []const usize, b: []const usize) bool {
    if (a.len != b.len) return false;

    for (a, b) |a_dim, b_dim| {
        if (a_dim != b_dim) return false;
    }

    return true;
}

fn isScalar(value: *const tensor.Tensor) bool {
    return value.elementCountSelf() == 1 and (value.shape.len == 0 or
        (value.shape.len == 1 and value.shape[0] == 1));
}

fn canBroadcastLastDim(target_shape: []const usize, bias_shape: []const usize) bool {
    if (target_shape.len == 0 or bias_shape.len != 1) return false;
    return bias_shape[0] == target_shape[target_shape.len - 1];
}

fn addLastDimBroadcast(
    allocator: std.mem.Allocator,
    target: *const tensor.Tensor,
    bias: *const tensor.Tensor,
) !tensor.Tensor {
    const target_data = try target.float32Data();
    const bias_data = try bias.float32Data();

    var output = try tensor.Tensor.initZeros(allocator, target.shape);
    errdefer output.deinit(allocator);
    const output_data = try output.float32DataMut();

    const width = bias.shape[0];

    for (output_data, 0..) |*value, index| {
        value.* = target_data[index] + bias_data[index % width];
    }

    return output;
}

fn nchwIndex(shape: []const usize, n: usize, c: usize, h: usize, w: usize) usize {
    return ((n * shape[1] + c) * shape[2] + h) * shape[3] + w;
}

fn normalizeAxis(axis: i64, rank: usize, allow_end: bool) !usize {
    const rank_i64 = std.math.cast(i64, rank) orelse return error.RankTooLarge;
    const normalized = if (axis < 0) axis + rank_i64 else axis;
    const max_axis = if (allow_end) rank_i64 else rank_i64 - 1;

    if (normalized < 0 or normalized > max_axis) return error.AxisOutOfBounds;

    return std.math.cast(usize, normalized) orelse error.AxisOutOfBounds;
}

fn matrixValue(data: []const f32, shape: []const usize, row: usize, col: usize, transposed: bool) f32 {
    const source_row = if (transposed) col else row;
    const source_col = if (transposed) row else col;
    return data[source_row * shape[1] + source_col];
}

fn biasValue(
    bias: *const tensor.Tensor,
    row: usize,
    col: usize,
    output_rows: usize,
    output_cols: usize,
) !f32 {
    const bias_data = try bias.float32Data();

    if (bias.shape.len == 0) return bias_data[0];

    if (bias.shape.len == 1) {
        if (bias.shape[0] == 1) return bias_data[0];
        if (bias.shape[0] == output_cols) return bias_data[col];
        if (bias.shape[0] == output_rows) return bias_data[row];
        return error.GemmBiasShapeMismatch;
    }

    if (bias.shape.len == 2) {
        const rows = bias.shape[0];
        const cols = bias.shape[1];

        if (rows == output_rows and cols == output_cols) return bias_data[row * cols + col];
        if (rows == 1 and cols == output_cols) return bias_data[col];
        if (rows == output_rows and cols == 1) return bias_data[row];
        if (rows == 1 and cols == 1) return bias_data[0];
    }

    return error.GemmBiasShapeMismatch;
}

fn tensorFromTensorProto(allocator: std.mem.Allocator, value: *const onnx.TensorProto) !tensor.Tensor {
    try requireFloat32(value.data_type);

    if (value.data_location) |location| {
        if (location != .DEFAULT) return error.ExternalTensorDataUnsupported;
    }

    if (value.segment != null) return error.TensorSegmentsUnsupported;

    const shape = try shapeFromDims(allocator, value.dims.items);
    errdefer allocator.free(shape);

    const count = try tensor.elementCount(shape);
    const data = try allocator.alloc(f32, count);
    errdefer allocator.free(data);

    if (value.raw_data) |raw_data| {
        try fillFloat32FromRawData(data, raw_data);
    } else {
        if (value.float_data.items.len != count) return error.TensorElementCountMismatch;
        @memcpy(data, value.float_data.items);
    }

    return .{
        .dtype = .float32,
        .shape = shape,
        .data = .{ .float32 = data },
    };
}

fn shapeFromDims(allocator: std.mem.Allocator, dims: []const i64) ![]usize {
    const shape = try allocator.alloc(usize, dims.len);
    errdefer allocator.free(shape);

    for (dims, 0..) |dim, index| {
        if (dim <= 0) return error.InvalidTensorDimension;
        shape[index] = std.math.cast(usize, dim) orelse return error.DimensionTooLarge;
    }

    return shape;
}

fn requireFloat32(data_type: ?i32) !void {
    const actual = data_type orelse return error.MissingTensorDataType;
    if (actual != @intFromEnum(onnx.TensorProto.DataType.FLOAT)) {
        return error.UnsupportedTensorDataType;
    }
}

fn fillFloat32FromRawData(out: []f32, raw_data: []const u8) !void {
    const expected_bytes = try std.math.mul(usize, out.len, @sizeOf(f32));
    if (raw_data.len != expected_bytes) return error.TensorRawDataLengthMismatch;

    for (out, 0..) |*value, index| {
        const offset = index * @sizeOf(f32);
        const bits =
            @as(u32, raw_data[offset]) |
            (@as(u32, raw_data[offset + 1]) << 8) |
            (@as(u32, raw_data[offset + 2]) << 16) |
            (@as(u32, raw_data[offset + 3]) << 24);
        value.* = @bitCast(bits);
    }
}

test "flatten reshapes tensor around axis" {
    const allocator = std.testing.allocator;

    const shape = [_]usize{ 2, 3, 4 };
    const data = [_]f32{
        0, 1, 2, 3,
        4, 5, 6, 7,
        8, 9, 10, 11,
        12, 13, 14, 15,
        16, 17, 18, 19,
        20, 21, 22, 23,
    };

    var input = try tensor.Tensor.init(allocator, &shape, &data);
    defer input.deinit(allocator);

    var output = try flatten(allocator, &input, 1);
    defer output.deinit(allocator);

    const expected_shape = [_]usize{ 2, 12 };
    const output_data = try output.float32Data();

    try std.testing.expectEqualSlices(usize, &expected_shape, output.shape);
    try std.testing.expectEqualSlices(f32, &data, output_data);
}

test "conv runs 2d nchw convolution without padding" {
    const allocator = std.testing.allocator;

    const x_shape = [_]usize{ 1, 1, 3, 3 };
    const x_data = [_]f32{
        1, 2, 3,
        4, 5, 6,
        7, 8, 9,
    };
    var x = try tensor.Tensor.init(allocator, &x_shape, &x_data);
    defer x.deinit(allocator);

    const w_shape = [_]usize{ 1, 1, 2, 2 };
    const w_data = [_]f32{
        1, 0,
        0, 1,
    };
    var w = try tensor.Tensor.init(allocator, &w_shape, &w_data);
    defer w.deinit(allocator);

    const b_shape = [_]usize{1};
    const b_data = [_]f32{0.5};
    var b = try tensor.Tensor.init(allocator, &b_shape, &b_data);
    defer b.deinit(allocator);

    var output = try conv(allocator, &x, &w, &b, .{});
    defer output.deinit(allocator);

    const expected_shape = [_]usize{ 1, 1, 2, 2 };
    const expected_data = [_]f32{
        6.5, 8.5,
        12.5, 14.5,
    };
    const output_data = try output.float32Data();

    try std.testing.expectEqualSlices(usize, &expected_shape, output.shape);
    try std.testing.expectEqualSlices(f32, &expected_data, output_data);
}

test "matmul multiplies two 2d float tensors" {
    const allocator = std.testing.allocator;

    const a_shape = [_]usize{ 2, 3 };
    const a_data = [_]f32{ 1, 2, 3, 4, 5, 6 };
    var a = try tensor.Tensor.init(allocator, &a_shape, &a_data);
    defer a.deinit(allocator);

    const b_shape = [_]usize{ 3, 2 };
    const b_data = [_]f32{ 7, 8, 9, 10, 11, 12 };
    var b = try tensor.Tensor.init(allocator, &b_shape, &b_data);
    defer b.deinit(allocator);

    var output = try matmul(allocator, &a, &b);
    defer output.deinit(allocator);

    const expected_shape = [_]usize{ 2, 2 };
    const expected_data = [_]f32{ 58, 64, 139, 154 };
    const output_data = try output.float32Data();

    try std.testing.expectEqualSlices(usize, &expected_shape, output.shape);
    try std.testing.expectEqualSlices(f32, &expected_data, output_data);
}
