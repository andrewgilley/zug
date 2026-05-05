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

pub const PoolOptions = struct {
    kernel_shape: [2]usize,
    pads: [4]usize = .{ 0, 0, 0, 0 },
    strides: [2]usize = .{ 1, 1 },
    ceil_mode: bool = false,
    count_include_pad: bool = false,
};

pub const BatchNormOptions = struct {
    epsilon: f32 = 0.00001,
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
    return elementwiseBinary(allocator, a, b, addFloat);
}

pub fn sub(
    allocator: std.mem.Allocator,
    a: *const tensor.Tensor,
    b: *const tensor.Tensor,
) !tensor.Tensor {
    return elementwiseBinary(allocator, a, b, subFloat);
}

pub fn mul(
    allocator: std.mem.Allocator,
    a: *const tensor.Tensor,
    b: *const tensor.Tensor,
) !tensor.Tensor {
    return elementwiseBinary(allocator, a, b, mulFloat);
}

pub fn div(
    allocator: std.mem.Allocator,
    a: *const tensor.Tensor,
    b: *const tensor.Tensor,
) !tensor.Tensor {
    return elementwiseBinary(allocator, a, b, divFloat);
}

pub fn relu(allocator: std.mem.Allocator, input: *const tensor.Tensor) !tensor.Tensor {
    return elementwiseUnary(allocator, input, reluFloat);
}

pub fn sigmoid(allocator: std.mem.Allocator, input: *const tensor.Tensor) !tensor.Tensor {
    return elementwiseUnary(allocator, input, sigmoidFloat);
}

pub fn tanh(allocator: std.mem.Allocator, input: *const tensor.Tensor) !tensor.Tensor {
    return elementwiseUnary(allocator, input, tanhFloat);
}

pub fn reshape(
    allocator: std.mem.Allocator,
    input: *const tensor.Tensor,
    shape_tensor: *const tensor.Tensor,
    allow_zero: bool,
) !tensor.Tensor {
    const target_shape = try reshapeTargetShape(allocator, input.shape, input.elementCountSelf(), shape_tensor, allow_zero);
    defer allocator.free(target_shape);

    return cloneWithShape(allocator, input, target_shape);
}

pub fn transpose(
    allocator: std.mem.Allocator,
    input: *const tensor.Tensor,
    maybe_perm: ?[]const usize,
) !tensor.Tensor {
    const input_data = try input.float32Data();
    const rank = input.shape.len;

    const perm = if (maybe_perm) |provided| provided else blk: {
        const generated = try allocator.alloc(usize, rank);
        for (generated, 0..) |*axis, index| {
            axis.* = rank - 1 - index;
        }
        break :blk generated;
    };
    defer if (maybe_perm == null) allocator.free(perm);

    try validatePermutation(perm, rank);

    const output_shape = try allocator.alloc(usize, rank);
    errdefer allocator.free(output_shape);
    for (output_shape, 0..) |*dim, index| {
        dim.* = input.shape[perm[index]];
    }

    const count = input.elementCountSelf();
    const output_data = try allocator.alloc(f32, count);
    errdefer allocator.free(output_data);

    const input_strides = try strides(allocator, input.shape);
    defer allocator.free(input_strides);

    const output_strides = try strides(allocator, output_shape);
    defer allocator.free(output_strides);

    const output_indices = try allocator.alloc(usize, rank);
    defer allocator.free(output_indices);

    const input_indices = try allocator.alloc(usize, rank);
    defer allocator.free(input_indices);

    for (output_data, 0..) |*value, output_index| {
        linearToIndices(output_index, output_shape, output_strides, output_indices);
        @memset(input_indices, 0);

        for (perm, 0..) |input_axis, output_axis| {
            input_indices[input_axis] = output_indices[output_axis];
        }

        const input_index = indicesToLinear(input_indices, input_strides);
        value.* = input_data[input_index];
    }

    return tensor.Tensor.initOwnedFloat32(allocator, output_shape, output_data);
}

pub fn clip(
    allocator: std.mem.Allocator,
    input: *const tensor.Tensor,
    min_input: ?*const tensor.Tensor,
    max_input: ?*const tensor.Tensor,
    min_attr: ?f32,
    max_attr: ?f32,
) !tensor.Tensor {
    const input_data = try input.float32Data();
    const min_value = if (min_input) |value| try scalarFloat32(value) else min_attr;
    const max_value = if (max_input) |value| try scalarFloat32(value) else max_attr;

    var output = try tensor.Tensor.initZeros(allocator, input.shape);
    errdefer output.deinit(allocator);
    const output_data = try output.float32DataMut();

    for (output_data, input_data) |*out, value| {
        var clipped = value;
        if (min_value) |minimum| {
            if (clipped < minimum) clipped = minimum;
        }
        if (max_value) |maximum| {
            if (clipped > maximum) clipped = maximum;
        }
        out.* = clipped;
    }

    return output;
}

pub fn globalAveragePool(allocator: std.mem.Allocator, input: *const tensor.Tensor) !tensor.Tensor {
    const input_data = try input.float32Data();
    if (input.shape.len < 3) return error.GlobalAveragePoolRequiresSpatialTensor;

    const batch = input.shape[0];
    const channels = input.shape[1];
    const spatial_count = try tensor.elementCount(input.shape[2..]);

    const output_shape = try allocator.dupe(usize, input.shape);
    defer allocator.free(output_shape);
    for (output_shape[2..]) |*dim| {
        dim.* = 1;
    }

    var output = try tensor.Tensor.initZeros(allocator, output_shape);
    errdefer output.deinit(allocator);
    const output_data = try output.float32DataMut();

    for (0..batch) |n| {
        for (0..channels) |c| {
            const base = (n * channels + c) * spatial_count;
            var sum: f32 = 0;

            for (0..spatial_count) |index| {
                sum += input_data[base + index];
            }

            output_data[n * channels + c] = sum / @as(f32, @floatFromInt(spatial_count));
        }
    }

    return output;
}

pub fn maxPool(
    allocator: std.mem.Allocator,
    input: *const tensor.Tensor,
    options: PoolOptions,
) !tensor.Tensor {
    const input_data = try input.float32Data();
    if (input.shape.len != 4) return error.PoolRequiresNchwTensor;
    if (options.ceil_mode) return error.PoolCeilModeUnsupported;

    const output_shape = try poolOutputShape(input.shape, options);
    var output = try tensor.Tensor.initZeros(allocator, &output_shape);
    errdefer output.deinit(allocator);
    const output_data = try output.float32DataMut();

    const batch = input.shape[0];
    const channels = input.shape[1];
    const input_h = input.shape[2];
    const input_w = input.shape[3];
    const output_h = output_shape[2];
    const output_w = output_shape[3];

    const kernel_h = options.kernel_shape[0];
    const kernel_w = options.kernel_shape[1];
    const stride_h = options.strides[0];
    const stride_w = options.strides[1];
    const pad_top = options.pads[0];
    const pad_left = options.pads[1];

    for (0..batch) |n| {
        for (0..channels) |c| {
            for (0..output_h) |oh| {
                for (0..output_w) |ow| {
                    var max_value = -std.math.inf(f32);
                    var has_value = false;

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

                            const value = input_data[nchwIndex(input.shape, n, c, input_y, input_x)];
                            if (!has_value or value > max_value) max_value = value;
                            has_value = true;
                        }
                    }

                    if (!has_value) return error.InvalidPoolWindow;
                    output_data[nchwIndex(output.shape, n, c, oh, ow)] = max_value;
                }
            }
        }
    }

    return output;
}

pub fn averagePool(
    allocator: std.mem.Allocator,
    input: *const tensor.Tensor,
    options: PoolOptions,
) !tensor.Tensor {
    const input_data = try input.float32Data();
    if (input.shape.len != 4) return error.PoolRequiresNchwTensor;
    if (options.ceil_mode) return error.PoolCeilModeUnsupported;

    const output_shape = try poolOutputShape(input.shape, options);
    var output = try tensor.Tensor.initZeros(allocator, &output_shape);
    errdefer output.deinit(allocator);
    const output_data = try output.float32DataMut();

    const batch = input.shape[0];
    const channels = input.shape[1];
    const input_h = input.shape[2];
    const input_w = input.shape[3];
    const output_h = output_shape[2];
    const output_w = output_shape[3];

    const kernel_h = options.kernel_shape[0];
    const kernel_w = options.kernel_shape[1];
    const stride_h = options.strides[0];
    const stride_w = options.strides[1];
    const pad_top = options.pads[0];
    const pad_left = options.pads[1];

    for (0..batch) |n| {
        for (0..channels) |c| {
            for (0..output_h) |oh| {
                for (0..output_w) |ow| {
                    var sum: f32 = 0;
                    var valid_count: usize = 0;

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

                            sum += input_data[nchwIndex(input.shape, n, c, input_y, input_x)];
                            valid_count += 1;
                        }
                    }

                    const divisor = if (options.count_include_pad)
                        kernel_h * kernel_w
                    else
                        valid_count;
                    if (divisor == 0) return error.InvalidPoolWindow;

                    output_data[nchwIndex(output.shape, n, c, oh, ow)] =
                        sum / @as(f32, @floatFromInt(divisor));
                }
            }
        }
    }

    return output;
}

pub fn batchNormalization(
    allocator: std.mem.Allocator,
    input: *const tensor.Tensor,
    scale: *const tensor.Tensor,
    bias: *const tensor.Tensor,
    mean: *const tensor.Tensor,
    variance: *const tensor.Tensor,
    options: BatchNormOptions,
) !tensor.Tensor {
    const input_data = try input.float32Data();
    const scale_data = try scale.float32Data();
    const bias_data = try bias.float32Data();
    const mean_data = try mean.float32Data();
    const variance_data = try variance.float32Data();

    if (input.shape.len < 2) return error.BatchNormRequiresChannelTensor;
    const channels = input.shape[1];
    if (scale.elementCountSelf() != channels or
        bias.elementCountSelf() != channels or
        mean.elementCountSelf() != channels or
        variance.elementCountSelf() != channels)
    {
        return error.BatchNormParameterShapeMismatch;
    }

    const inner = try tensor.elementCount(input.shape[2..]);
    var output = try tensor.Tensor.initZeros(allocator, input.shape);
    errdefer output.deinit(allocator);
    const output_data = try output.float32DataMut();

    for (output_data, input_data, 0..) |*out, value, index| {
        const channel = (index / inner) % channels;
        out.* = ((value - mean_data[channel]) / @sqrt(variance_data[channel] + options.epsilon)) *
            scale_data[channel] + bias_data[channel];
    }

    return output;
}

pub fn shapeTensor(allocator: std.mem.Allocator, input: *const tensor.Tensor) !tensor.Tensor {
    const output_shape = try allocator.alloc(usize, 1);
    errdefer allocator.free(output_shape);
    output_shape[0] = input.shape.len;

    const data = try allocator.alloc(i64, input.shape.len);
    errdefer allocator.free(data);

    for (data, input.shape) |*out, dim| {
        out.* = std.math.cast(i64, dim) orelse return error.DimensionTooLarge;
    }

    return tensor.Tensor.initOwnedInt64(allocator, output_shape, data);
}

pub fn gather(
    allocator: std.mem.Allocator,
    data: *const tensor.Tensor,
    indices: *const tensor.Tensor,
    axis: i64,
) !tensor.Tensor {
    const normalized_axis = try normalizeAxis(axis, data.shape.len, false);
    if (data.shape.len != 1 or normalized_axis != 0) return error.GatherOnlySupportsAxisZeroVectors;

    return switch (data.data) {
        .float32 => |items| gatherVector(f32, allocator, items, indices, tensor.Tensor.initOwnedFloat32),
        .int64 => |items| gatherVector(i64, allocator, items, indices, tensor.Tensor.initOwnedInt64),
        .int32 => |items| gatherVector(i32, allocator, items, indices, tensor.Tensor.initOwnedInt32),
        else => error.GatherUnsupportedDataType,
    };
}

pub fn unsqueeze(
    allocator: std.mem.Allocator,
    input: *const tensor.Tensor,
    axes: []const i64,
) !tensor.Tensor {
    const output_rank = try std.math.add(usize, input.shape.len, axes.len);
    var axis_mask = [_]bool{false} ** 16;
    if (output_rank > axis_mask.len) return error.UnsqueezeRankTooLarge;

    for (axes) |axis| {
        const normalized = try normalizeAxis(axis, output_rank, false);
        if (axis_mask[normalized]) return error.UnsqueezeDuplicateAxis;
        axis_mask[normalized] = true;
    }

    const output_shape = try allocator.alloc(usize, output_rank);
    defer allocator.free(output_shape);

    var input_axis: usize = 0;
    for (output_shape, 0..) |*dim, output_axis| {
        if (axis_mask[output_axis]) {
            dim.* = 1;
        } else {
            dim.* = input.shape[input_axis];
            input_axis += 1;
        }
    }

    return cloneWithShape(allocator, input, output_shape);
}

pub fn concat(
    allocator: std.mem.Allocator,
    inputs: []const *const tensor.Tensor,
    axis: i64,
) !tensor.Tensor {
    if (inputs.len == 0) return error.ConcatRequiresInput;

    const dtype = inputs[0].dtype;
    const rank = inputs[0].shape.len;
    const normalized_axis = try normalizeAxis(axis, rank, false);
    const output_shape = try allocator.dupe(usize, inputs[0].shape);
    errdefer allocator.free(output_shape);
    output_shape[normalized_axis] = 0;

    for (inputs) |input| {
        if (input.dtype != dtype) return error.ConcatDTypeMismatch;
        if (input.shape.len != rank) return error.ConcatRankMismatch;

        for (input.shape, 0..) |dim, index| {
            if (index == normalized_axis) continue;
            if (dim != output_shape[index]) return error.ConcatShapeMismatch;
        }

        output_shape[normalized_axis] = try std.math.add(usize, output_shape[normalized_axis], input.shape[normalized_axis]);
    }

    return switch (dtype) {
        .float32 => try concatTyped(f32, allocator, inputs, output_shape, normalized_axis, tensor.Tensor.initOwnedFloat32),
        .int64 => try concatTyped(i64, allocator, inputs, output_shape, normalized_axis, tensor.Tensor.initOwnedInt64),
        .int32 => try concatTyped(i32, allocator, inputs, output_shape, normalized_axis, tensor.Tensor.initOwnedInt32),
        else => error.ConcatUnsupportedDType,
    };
}

pub fn squeeze(
    allocator: std.mem.Allocator,
    input: *const tensor.Tensor,
    maybe_axes: ?[]const i64,
) !tensor.Tensor {
    var squeeze_mask = [_]bool{false} ** 16;
    if (input.shape.len > squeeze_mask.len) return error.SqueezeRankTooLarge;

    if (maybe_axes) |axes| {
        for (axes) |axis| {
            const normalized = try normalizeAxis(axis, input.shape.len, false);
            if (input.shape[normalized] != 1) return error.SqueezeDimensionNotOne;
            squeeze_mask[normalized] = true;
        }
    } else {
        for (input.shape, 0..) |dim, index| {
            squeeze_mask[index] = dim == 1;
        }
    }

    var output_rank: usize = 0;
    for (input.shape, 0..) |_, index| {
        if (!squeeze_mask[index]) output_rank += 1;
    }

    const output_shape = try allocator.alloc(usize, output_rank);
    defer allocator.free(output_shape);

    var output_index: usize = 0;
    for (input.shape, 0..) |dim, index| {
        if (squeeze_mask[index]) continue;
        output_shape[output_index] = dim;
        output_index += 1;
    }

    return cloneWithShape(allocator, input, output_shape);
}

pub fn cast(
    allocator: std.mem.Allocator,
    input: *const tensor.Tensor,
    dtype: tensor.DType,
) !tensor.Tensor {
    return switch (dtype) {
        .float32 => castToFloat32(allocator, input),
        .int64 => castToInt64(allocator, input),
        .int32 => castToInt32(allocator, input),
        .uint8 => castToUint8(allocator, input),
        .bool => castToBool(allocator, input),
    };
}

pub fn slice(
    allocator: std.mem.Allocator,
    data: *const tensor.Tensor,
    starts_tensor: *const tensor.Tensor,
    ends_tensor: *const tensor.Tensor,
    axes_tensor: ?*const tensor.Tensor,
    steps_tensor: ?*const tensor.Tensor,
) !tensor.Tensor {
    const starts = try tensorToI64List(allocator, starts_tensor);
    defer allocator.free(starts);

    const ends = try tensorToI64List(allocator, ends_tensor);
    defer allocator.free(ends);

    if (starts.len != ends.len) return error.SliceParameterLengthMismatch;

    const axes = if (axes_tensor) |value|
        try tensorToI64List(allocator, value)
    else blk: {
        const generated = try allocator.alloc(i64, starts.len);
        for (generated, 0..) |*axis, index| {
            axis.* = std.math.cast(i64, index) orelse return error.RankTooLarge;
        }
        break :blk generated;
    };
    defer allocator.free(axes);

    const steps = if (steps_tensor) |value|
        try tensorToI64List(allocator, value)
    else blk: {
        const generated = try allocator.alloc(i64, starts.len);
        @memset(generated, 1);
        break :blk generated;
    };
    defer allocator.free(steps);

    if (axes.len != starts.len or steps.len != starts.len) {
        return error.SliceParameterLengthMismatch;
    }

    return switch (data.data) {
        .float32 => |items| sliceTyped(f32, allocator, data.shape, items, starts, ends, axes, steps, tensor.Tensor.initOwnedFloat32),
        .int64 => |items| sliceTyped(i64, allocator, data.shape, items, starts, ends, axes, steps, tensor.Tensor.initOwnedInt64),
        .int32 => |items| sliceTyped(i32, allocator, data.shape, items, starts, ends, axes, steps, tensor.Tensor.initOwnedInt32),
        .uint8 => |items| sliceTyped(u8, allocator, data.shape, items, starts, ends, axes, steps, tensor.Tensor.initOwnedUint8),
        .bool => |items| sliceTyped(bool, allocator, data.shape, items, starts, ends, axes, steps, tensor.Tensor.initOwnedBool),
    };
}

fn poolOutputShape(input_shape: []const usize, options: PoolOptions) ![4]usize {
    const kernel_h = options.kernel_shape[0];
    const kernel_w = options.kernel_shape[1];
    const stride_h = options.strides[0];
    const stride_w = options.strides[1];

    if (kernel_h == 0 or kernel_w == 0) return error.PoolInvalidKernel;
    if (stride_h == 0 or stride_w == 0) return error.PoolInvalidStride;

    return .{
        input_shape[0],
        input_shape[1],
        try poolOutputDim(input_shape[2], options.pads[0], options.pads[2], kernel_h, stride_h),
        try poolOutputDim(input_shape[3], options.pads[1], options.pads[3], kernel_w, stride_w),
    };
}

fn poolOutputDim(input_dim: usize, pad_before: usize, pad_after: usize, kernel: usize, stride: usize) !usize {
    const padded = try std.math.add(usize, try std.math.add(usize, input_dim, pad_before), pad_after);
    if (padded < kernel) return error.PoolKernelLargerThanInput;

    return ((padded - kernel) / stride) + 1;
}

fn castToFloat32(allocator: std.mem.Allocator, input: *const tensor.Tensor) !tensor.Tensor {
    const data = try allocator.alloc(f32, input.elementCountSelf());
    errdefer allocator.free(data);

    switch (input.data) {
        .float32 => |items| @memcpy(data, items),
        .int64 => |items| {
            for (data, items) |*out, item| out.* = @floatFromInt(item);
        },
        .int32 => |items| {
            for (data, items) |*out, item| out.* = @floatFromInt(item);
        },
        .uint8 => |items| {
            for (data, items) |*out, item| out.* = @floatFromInt(item);
        },
        .bool => |items| {
            for (data, items) |*out, item| out.* = if (item) 1 else 0;
        },
    }

    const output_shape = try allocator.dupe(usize, input.shape);
    errdefer allocator.free(output_shape);

    return tensor.Tensor.initOwnedFloat32(allocator, output_shape, data);
}

fn castToInt64(allocator: std.mem.Allocator, input: *const tensor.Tensor) !tensor.Tensor {
    const data = try allocator.alloc(i64, input.elementCountSelf());
    errdefer allocator.free(data);

    switch (input.data) {
        .float32 => |items| {
            for (data, items) |*out, item| out.* = @intFromFloat(item);
        },
        .int64 => |items| @memcpy(data, items),
        .int32 => |items| {
            for (data, items) |*out, item| out.* = item;
        },
        .uint8 => |items| {
            for (data, items) |*out, item| out.* = item;
        },
        .bool => |items| {
            for (data, items) |*out, item| out.* = if (item) 1 else 0;
        },
    }

    const output_shape = try allocator.dupe(usize, input.shape);
    errdefer allocator.free(output_shape);

    return tensor.Tensor.initOwnedInt64(allocator, output_shape, data);
}

fn castToInt32(allocator: std.mem.Allocator, input: *const tensor.Tensor) !tensor.Tensor {
    const data = try allocator.alloc(i32, input.elementCountSelf());
    errdefer allocator.free(data);

    switch (input.data) {
        .float32 => |items| {
            for (data, items) |*out, item| out.* = @intFromFloat(item);
        },
        .int64 => |items| {
            for (data, items) |*out, item| out.* = std.math.cast(i32, item) orelse return error.CastOutOfRange;
        },
        .int32 => |items| @memcpy(data, items),
        .uint8 => |items| {
            for (data, items) |*out, item| out.* = item;
        },
        .bool => |items| {
            for (data, items) |*out, item| out.* = if (item) 1 else 0;
        },
    }

    const output_shape = try allocator.dupe(usize, input.shape);
    errdefer allocator.free(output_shape);

    return tensor.Tensor.initOwnedInt32(allocator, output_shape, data);
}

fn castToUint8(allocator: std.mem.Allocator, input: *const tensor.Tensor) !tensor.Tensor {
    const data = try allocator.alloc(u8, input.elementCountSelf());
    errdefer allocator.free(data);

    switch (input.data) {
        .float32 => |items| {
            for (data, items) |*out, item| out.* = @intFromFloat(item);
        },
        .int64 => |items| {
            for (data, items) |*out, item| out.* = std.math.cast(u8, item) orelse return error.CastOutOfRange;
        },
        .int32 => |items| {
            for (data, items) |*out, item| out.* = std.math.cast(u8, item) orelse return error.CastOutOfRange;
        },
        .uint8 => |items| @memcpy(data, items),
        .bool => |items| {
            for (data, items) |*out, item| out.* = if (item) 1 else 0;
        },
    }

    const output_shape = try allocator.dupe(usize, input.shape);
    errdefer allocator.free(output_shape);

    return tensor.Tensor.initOwnedUint8(allocator, output_shape, data);
}

fn castToBool(allocator: std.mem.Allocator, input: *const tensor.Tensor) !tensor.Tensor {
    const data = try allocator.alloc(bool, input.elementCountSelf());
    errdefer allocator.free(data);

    switch (input.data) {
        .float32 => |items| {
            for (data, items) |*out, item| out.* = item != 0;
        },
        .int64 => |items| {
            for (data, items) |*out, item| out.* = item != 0;
        },
        .int32 => |items| {
            for (data, items) |*out, item| out.* = item != 0;
        },
        .uint8 => |items| {
            for (data, items) |*out, item| out.* = item != 0;
        },
        .bool => |items| @memcpy(data, items),
    }

    const output_shape = try allocator.dupe(usize, input.shape);
    errdefer allocator.free(output_shape);

    return tensor.Tensor.initOwnedBool(allocator, output_shape, data);
}

fn tensorToI64List(allocator: std.mem.Allocator, value: *const tensor.Tensor) ![]i64 {
    return switch (value.data) {
        .int64 => |items| allocator.dupe(i64, items),
        .int32 => |items| blk: {
            const out = try allocator.alloc(i64, items.len);
            for (out, items) |*target, item| {
                target.* = item;
            }
            break :blk out;
        },
        else => error.ExpectedIntegerTensor,
    };
}

fn sliceTyped(
    comptime T: type,
    allocator: std.mem.Allocator,
    input_shape: []const usize,
    input_data: []const T,
    starts: []const i64,
    ends: []const i64,
    axes: []const i64,
    steps: []const i64,
    comptime initFn: anytype,
) !tensor.Tensor {
    const rank = input_shape.len;
    const output_shape = try allocator.dupe(usize, input_shape);
    errdefer allocator.free(output_shape);

    const start_by_axis = try allocator.alloc(usize, rank);
    defer allocator.free(start_by_axis);
    @memset(start_by_axis, 0);

    const step_by_axis = try allocator.alloc(usize, rank);
    defer allocator.free(step_by_axis);
    @memset(step_by_axis, 1);

    for (starts, ends, axes, steps) |start_raw, end_raw, axis_raw, step_raw| {
        if (step_raw <= 0) return error.SliceStepUnsupported;

        const axis = try normalizeAxis(axis_raw, rank, false);
        const step = std.math.cast(usize, step_raw) orelse return error.SliceStepUnsupported;
        const start = try clampSliceIndex(start_raw, input_shape[axis]);
        const end = try clampSliceIndex(end_raw, input_shape[axis]);

        const output_dim = if (end <= start)
            0
        else
            ((end - start + step - 1) / step);
        if (output_dim == 0) return error.SliceEmptyOutputUnsupported;

        output_shape[axis] = output_dim;
        start_by_axis[axis] = start;
        step_by_axis[axis] = step;
    }

    const output_count = try tensor.elementCount(output_shape);
    const output_data = try allocator.alloc(T, output_count);
    errdefer allocator.free(output_data);

    const input_strides = try strides(allocator, input_shape);
    defer allocator.free(input_strides);

    const output_strides = try strides(allocator, output_shape);
    defer allocator.free(output_strides);

    const output_indices = try allocator.alloc(usize, rank);
    defer allocator.free(output_indices);

    const input_indices = try allocator.alloc(usize, rank);
    defer allocator.free(input_indices);

    for (output_data, 0..) |*out, output_linear| {
        linearToIndices(output_linear, output_shape, output_strides, output_indices);

        for (input_indices, 0..) |*input_index, axis| {
            input_index.* = start_by_axis[axis] + output_indices[axis] * step_by_axis[axis];
        }

        out.* = input_data[indicesToLinear(input_indices, input_strides)];
    }

    return initFn(allocator, output_shape, output_data);
}

fn clampSliceIndex(raw_index: i64, dim: usize) !usize {
    const dim_i64 = std.math.cast(i64, dim) orelse return error.DimensionTooLarge;
    var index = if (raw_index < 0) raw_index + dim_i64 else raw_index;

    if (index < 0) index = 0;
    if (index > dim_i64) index = dim_i64;

    return std.math.cast(usize, index) orelse error.DimensionTooLarge;
}

fn gatherVector(
    comptime T: type,
    allocator: std.mem.Allocator,
    items: []const T,
    indices: *const tensor.Tensor,
    comptime initFn: anytype,
) !tensor.Tensor {
    const output_shape = try allocator.dupe(usize, indices.shape);
    errdefer allocator.free(output_shape);

    const output_count = indices.elementCountSelf();
    const output_data = try allocator.alloc(T, output_count);
    errdefer allocator.free(output_data);

    for (output_data, 0..) |*out, index| {
        const raw_index = try gatherIndex(indices, index);
        const normalized = if (raw_index < 0)
            raw_index + @as(i64, @intCast(items.len))
        else
            raw_index;

        if (normalized < 0) return error.GatherIndexOutOfBounds;
        const actual = std.math.cast(usize, normalized) orelse return error.GatherIndexOutOfBounds;
        if (actual >= items.len) return error.GatherIndexOutOfBounds;

        out.* = items[actual];
    }

    return initFn(allocator, output_shape, output_data);
}

fn gatherIndex(indices: *const tensor.Tensor, index: usize) !i64 {
    return switch (indices.data) {
        .int64 => |items| items[index],
        .int32 => |items| items[index],
        else => error.GatherIndicesMustBeInteger,
    };
}

fn concatTyped(
    comptime T: type,
    allocator: std.mem.Allocator,
    inputs: []const *const tensor.Tensor,
    output_shape: []usize,
    axis: usize,
    comptime initFn: anytype,
) !tensor.Tensor {
    const output_count = try tensor.elementCount(output_shape);
    const output_data = try allocator.alloc(T, output_count);
    errdefer allocator.free(output_data);

    const outer = try tensor.elementCount(output_shape[0..axis]);
    const inner = try tensor.elementCount(output_shape[axis + 1 ..]);
    var output_offset: usize = 0;

    for (0..outer) |outer_index| {
        for (inputs) |input| {
            const input_data = typedData(T, input) catch return error.ConcatUnsupportedDType;
            const axis_len = input.shape[axis];
            const chunk_len = axis_len * inner;
            const input_offset = outer_index * chunk_len;

            @memcpy(output_data[output_offset..][0..chunk_len], input_data[input_offset..][0..chunk_len]);
            output_offset += chunk_len;
        }
    }

    return initFn(allocator, output_shape, output_data);
}

fn typedData(comptime T: type, value: *const tensor.Tensor) ![]const T {
    return switch (value.data) {
        .float32 => |items| if (T == f32) items else error.ConcatUnsupportedDType,
        .int64 => |items| if (T == i64) items else error.ConcatUnsupportedDType,
        .int32 => |items| if (T == i32) items else error.ConcatUnsupportedDType,
        else => error.ConcatUnsupportedDType,
    };
}

fn elementwiseBinary(
    allocator: std.mem.Allocator,
    a: *const tensor.Tensor,
    b: *const tensor.Tensor,
    op: *const fn (f32, f32) f32,
) !tensor.Tensor {
    const a_data = try a.float32Data();
    const b_data = try b.float32Data();

    if (sameShape(a.shape, b.shape)) {
        var output = try tensor.Tensor.initZeros(allocator, a.shape);
        errdefer output.deinit(allocator);
        const output_data = try output.float32DataMut();

        for (output_data, 0..) |*value, index| {
            value.* = op(a_data[index], b_data[index]);
        }

        return output;
    }

    if (isScalar(a)) {
        var output = try tensor.Tensor.initZeros(allocator, b.shape);
        errdefer output.deinit(allocator);
        const output_data = try output.float32DataMut();

        for (output_data, 0..) |*value, index| {
            value.* = op(a_data[0], b_data[index]);
        }

        return output;
    }

    if (isScalar(b)) {
        var output = try tensor.Tensor.initZeros(allocator, a.shape);
        errdefer output.deinit(allocator);
        const output_data = try output.float32DataMut();

        for (output_data, 0..) |*value, index| {
            value.* = op(a_data[index], b_data[0]);
        }

        return output;
    }

    if (canBroadcastLastDim(a.shape, b.shape)) {
        return binaryLastDimBroadcast(allocator, a, b, op);
    }

    if (canBroadcastLastDim(b.shape, a.shape)) {
        return binaryLastDimBroadcastSwapped(allocator, a, b, op);
    }

    return error.ElementwiseShapeMismatch;
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
    if (options.group == 0) return error.InvalidConvGroup;
    if (options.dilations[0] != 1 or options.dilations[1] != 1) return error.ConvDilationUnsupported;

    const batch = x.shape[0];
    const in_channels = x.shape[1];
    const input_h = x.shape[2];
    const input_w = x.shape[3];

    const out_channels = w.shape[0];
    const kernel_in_channels = w.shape[1];
    const kernel_h = w.shape[2];
    const kernel_w = w.shape[3];

    if (in_channels % options.group != 0) return error.ConvChannelMismatch;
    if (out_channels % options.group != 0) return error.ConvChannelMismatch;

    const in_channels_per_group = in_channels / options.group;
    const out_channels_per_group = out_channels / options.group;

    if (kernel_in_channels != in_channels_per_group) return error.ConvChannelMismatch;
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

    const output_shape = try allocator.alloc(usize, 4);
    errdefer allocator.free(output_shape);
    output_shape[0] = batch;
    output_shape[1] = out_channels;
    output_shape[2] = output_h;
    output_shape[3] = output_w;

    const output_count = try tensor.elementCount(output_shape);
    const output_data = try allocator.alloc(f32, output_count);
    errdefer allocator.free(output_data);

    if (canUseConv1x1NchwFastPath(options, kernel_h, kernel_w, input_h, input_w, output_h, output_w)) {
        conv1x1Nchw(x.shape, w.shape, x_data, w_data, bias_data, options, output_data);
        return tensor.Tensor.initOwnedFloat32(allocator, output_shape, output_data);
    }

    if (canUseDepthwiseConvFastPath(options, in_channels, kernel_in_channels)) {
        convDepthwiseNchw(x.shape, w.shape, x_data, w_data, bias_data, options, output_h, output_w, output_data);
        return tensor.Tensor.initOwnedFloat32(allocator, output_shape, output_data);
    }

    const input_spatial = input_h * input_w;
    const output_spatial = output_h * output_w;
    const kernel_spatial = kernel_h * kernel_w;

    for (0..batch) |n| {
        const input_batch_base = n * in_channels * input_spatial;
        const output_batch_base = n * out_channels * output_spatial;

        for (0..options.group) |group_index| {
            for (0..out_channels_per_group) |oc_in_group| {
                const oc = group_index * out_channels_per_group + oc_in_group;
                const output_channel_base = output_batch_base + oc * output_spatial;

                for (0..output_h) |oh| {
                    for (0..output_w) |ow| {
                        var sum: f32 = if (bias_data) |bias| bias[oc] else 0.0;

                        for (0..in_channels_per_group) |ic_in_group| {
                            const ic = group_index * in_channels_per_group + ic_in_group;
                            const input_channel_base = input_batch_base + ic * input_spatial;
                            const weight_channel_base = (oc * kernel_in_channels + ic_in_group) * kernel_spatial;

                            for (0..kernel_h) |kh| {
                                const padded_y = oh * stride_h + kh;
                                if (padded_y < pad_top) continue;

                                const input_y = padded_y - pad_top;
                                if (input_y >= input_h) continue;
                                const input_row_base = input_channel_base + input_y * input_w;
                                const weight_row_base = weight_channel_base + kh * kernel_w;

                                for (0..kernel_w) |kw| {
                                    const padded_x = ow * stride_w + kw;
                                    if (padded_x < pad_left) continue;

                                    const input_x = padded_x - pad_left;
                                    if (input_x >= input_w) continue;

                                    sum += x_data[input_row_base + input_x] *
                                        w_data[weight_row_base + kw];
                                }
                            }
                        }

                        output_data[output_channel_base + oh * output_w + ow] = sum;
                    }
                }
            }
        }
    }

    return tensor.Tensor.initOwnedFloat32(allocator, output_shape, output_data);
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

    if (a.shape.len != 2 or b.shape.len != 2) return error.MatMulRequiresMatrices;

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

fn scalarFloat32(value: *const tensor.Tensor) !f32 {
    if (!isScalar(value)) return error.ExpectedScalarTensor;
    const data = try value.float32Data();
    return data[0];
}

fn canBroadcastLastDim(target_shape: []const usize, bias_shape: []const usize) bool {
    if (target_shape.len == 0 or bias_shape.len != 1) return false;
    return bias_shape[0] == target_shape[target_shape.len - 1];
}

fn elementwiseUnary(
    allocator: std.mem.Allocator,
    input: *const tensor.Tensor,
    op: *const fn (f32) f32,
) !tensor.Tensor {
    const input_data = try input.float32Data();
    var output = try tensor.Tensor.initZeros(allocator, input.shape);
    errdefer output.deinit(allocator);
    const output_data = try output.float32DataMut();

    for (output_data, input_data) |*out, value| {
        out.* = op(value);
    }

    return output;
}

fn binaryLastDimBroadcast(
    allocator: std.mem.Allocator,
    target: *const tensor.Tensor,
    bias: *const tensor.Tensor,
    op: *const fn (f32, f32) f32,
) !tensor.Tensor {
    const target_data = try target.float32Data();
    const bias_data = try bias.float32Data();

    var output = try tensor.Tensor.initZeros(allocator, target.shape);
    errdefer output.deinit(allocator);
    const output_data = try output.float32DataMut();

    const width = bias.shape[0];

    for (output_data, 0..) |*value, index| {
        value.* = op(target_data[index], bias_data[index % width]);
    }

    return output;
}

fn binaryLastDimBroadcastSwapped(
    allocator: std.mem.Allocator,
    a: *const tensor.Tensor,
    b: *const tensor.Tensor,
    op: *const fn (f32, f32) f32,
) !tensor.Tensor {
    const a_data = try a.float32Data();
    const b_data = try b.float32Data();

    var output = try tensor.Tensor.initZeros(allocator, b.shape);
    errdefer output.deinit(allocator);
    const output_data = try output.float32DataMut();

    const width = a.shape[0];

    for (output_data, 0..) |*value, index| {
        value.* = op(a_data[index % width], b_data[index]);
    }

    return output;
}

fn addFloat(a: f32, b: f32) f32 {
    return a + b;
}

fn subFloat(a: f32, b: f32) f32 {
    return a - b;
}

fn mulFloat(a: f32, b: f32) f32 {
    return a * b;
}

fn divFloat(a: f32, b: f32) f32 {
    return a / b;
}

fn reluFloat(value: f32) f32 {
    return if (value < 0) 0 else value;
}

fn sigmoidFloat(value: f32) f32 {
    return 1.0 / (1.0 + @exp(-value));
}

fn tanhFloat(value: f32) f32 {
    const doubled = 2.0 * value;
    return (@exp(doubled) - 1.0) / (@exp(doubled) + 1.0);
}

fn reshapeTargetShape(
    allocator: std.mem.Allocator,
    input_shape: []const usize,
    input_count: usize,
    shape_tensor: *const tensor.Tensor,
    allow_zero: bool,
) ![]usize {
    const raw_shape = switch (shape_tensor.data) {
        .int64 => |items| blk: {
            const out = try allocator.alloc(i64, items.len);
            @memcpy(out, items);
            break :blk out;
        },
        .int32 => |items| blk: {
            const out = try allocator.alloc(i64, items.len);
            for (out, items) |*value, item| {
                value.* = item;
            }
            break :blk out;
        },
        else => return error.ReshapeShapeTensorMustBeInteger,
    };
    defer allocator.free(raw_shape);

    const output_shape = try allocator.alloc(usize, raw_shape.len);
    errdefer allocator.free(output_shape);

    var known_count: usize = 1;
    var infer_index: ?usize = null;

    for (raw_shape, 0..) |dim, index| {
        if (dim == -1) {
            if (infer_index != null) return error.ReshapeMultipleInferredDimensions;
            infer_index = index;
            output_shape[index] = 1;
            continue;
        }

        if (dim == 0 and !allow_zero) {
            if (index >= input_shape.len) return error.ReshapeZeroDimensionOutOfBounds;
            output_shape[index] = input_shape[index];
        } else {
            if (dim <= 0) return error.InvalidTensorDimension;
            output_shape[index] = std.math.cast(usize, dim) orelse return error.DimensionTooLarge;
        }

        known_count = try std.math.mul(usize, known_count, output_shape[index]);
    }

    if (infer_index) |index| {
        if (known_count == 0 or input_count % known_count != 0) return error.ReshapeElementCountMismatch;
        output_shape[index] = input_count / known_count;
    } else if (known_count != input_count) {
        return error.ReshapeElementCountMismatch;
    }

    return output_shape;
}

fn cloneWithShape(
    allocator: std.mem.Allocator,
    input: *const tensor.Tensor,
    shape: []const usize,
) !tensor.Tensor {
    return switch (input.data) {
        .float32 => |items| tensor.Tensor.initFloat32(allocator, shape, items),
        .int64 => |items| tensor.Tensor.initInt64(allocator, shape, items),
        .int32 => |items| tensor.Tensor.initInt32(allocator, shape, items),
        .uint8 => |items| tensor.Tensor.initUint8(allocator, shape, items),
        .bool => |items| tensor.Tensor.initBool(allocator, shape, items),
    };
}

fn validatePermutation(perm: []const usize, rank: usize) !void {
    if (perm.len != rank) return error.TransposePermutationRankMismatch;

    var seen = [_]bool{false} ** 16;
    if (rank > seen.len) return error.TransposeRankTooLarge;

    for (perm) |axis| {
        if (axis >= rank) return error.TransposePermutationOutOfBounds;
        if (seen[axis]) return error.TransposePermutationDuplicateAxis;
        seen[axis] = true;
    }
}

fn strides(allocator: std.mem.Allocator, shape: []const usize) ![]usize {
    const out = try allocator.alloc(usize, shape.len);
    errdefer allocator.free(out);

    var stride: usize = 1;
    var index = shape.len;
    while (index > 0) {
        index -= 1;
        out[index] = stride;
        stride = try std.math.mul(usize, stride, shape[index]);
    }

    return out;
}

fn linearToIndices(
    linear_index: usize,
    shape: []const usize,
    shape_strides: []const usize,
    out: []usize,
) void {
    var remainder = linear_index;
    for (out, 0..) |*value, index| {
        value.* = remainder / shape_strides[index];
        remainder %= shape_strides[index];
        std.debug.assert(value.* < shape[index]);
    }
}

fn indicesToLinear(indices: []const usize, shape_strides: []const usize) usize {
    var linear: usize = 0;
    for (indices, shape_strides) |index, stride| {
        linear += index * stride;
    }
    return linear;
}

fn nchwIndex(shape: []const usize, n: usize, c: usize, h: usize, w: usize) usize {
    return ((n * shape[1] + c) * shape[2] + h) * shape[3] + w;
}

fn canUseConv1x1NchwFastPath(
    options: ConvOptions,
    kernel_h: usize,
    kernel_w: usize,
    input_h: usize,
    input_w: usize,
    output_h: usize,
    output_w: usize,
) bool {
    return kernel_h == 1 and
        kernel_w == 1 and
        options.pads[0] == 0 and
        options.pads[1] == 0 and
        options.pads[2] == 0 and
        options.pads[3] == 0 and
        options.strides[0] == 1 and
        options.strides[1] == 1 and
        output_h == input_h and
        output_w == input_w;
}

fn canUseDepthwiseConvFastPath(
    options: ConvOptions,
    in_channels: usize,
    kernel_in_channels: usize,
) bool {
    return options.group == in_channels and kernel_in_channels == 1;
}

fn conv1x1Nchw(
    x_shape: []const usize,
    w_shape: []const usize,
    x_data: []const f32,
    w_data: []const f32,
    bias_data: ?[]const f32,
    options: ConvOptions,
    output_data: []f32,
) void {
    const batch = x_shape[0];
    const in_channels = x_shape[1];
    const spatial = x_shape[2] * x_shape[3];
    const out_channels = w_shape[0];
    const in_channels_per_group = in_channels / options.group;
    const out_channels_per_group = out_channels / options.group;

    for (0..batch) |n| {
        const input_batch_base = n * in_channels * spatial;
        const output_batch_base = n * out_channels * spatial;

        for (0..options.group) |group_index| {
            const input_group_base = group_index * in_channels_per_group;

            for (0..out_channels_per_group) |oc_in_group| {
                const oc = group_index * out_channels_per_group + oc_in_group;
                const output_base = output_batch_base + oc * spatial;

                if (bias_data) |bias| {
                    @memset(output_data[output_base..][0..spatial], bias[oc]);
                } else {
                    @memset(output_data[output_base..][0..spatial], 0);
                }

                const weight_base = oc * in_channels_per_group;
                for (0..in_channels_per_group) |ic_in_group| {
                    const ic = input_group_base + ic_in_group;
                    const input_base = input_batch_base + ic * spatial;
                    const weight = w_data[weight_base + ic_in_group];

                    for (0..spatial) |spatial_index| {
                        output_data[output_base + spatial_index] +=
                            x_data[input_base + spatial_index] * weight;
                    }
                }
            }
        }
    }
}

fn convDepthwiseNchw(
    x_shape: []const usize,
    w_shape: []const usize,
    x_data: []const f32,
    w_data: []const f32,
    bias_data: ?[]const f32,
    options: ConvOptions,
    output_h: usize,
    output_w: usize,
    output_data: []f32,
) void {
    const batch = x_shape[0];
    const in_channels = x_shape[1];
    const input_h = x_shape[2];
    const input_w = x_shape[3];
    const out_channels = w_shape[0];
    const kernel_h = w_shape[2];
    const kernel_w = w_shape[3];
    const out_channels_per_group = out_channels / in_channels;
    const input_spatial = input_h * input_w;
    const output_spatial = output_h * output_w;
    const kernel_spatial = kernel_h * kernel_w;
    const stride_h = options.strides[0];
    const stride_w = options.strides[1];
    const pad_top = options.pads[0];
    const pad_left = options.pads[1];

    for (0..batch) |n| {
        const input_batch_base = n * in_channels * input_spatial;
        const output_batch_base = n * out_channels * output_spatial;

        for (0..in_channels) |channel| {
            const input_channel_base = input_batch_base + channel * input_spatial;

            for (0..out_channels_per_group) |oc_in_group| {
                const oc = channel * out_channels_per_group + oc_in_group;
                const output_channel_base = output_batch_base + oc * output_spatial;
                const weight_base = oc * kernel_spatial;

                for (0..output_h) |oh| {
                    const output_row_base = output_channel_base + oh * output_w;

                    for (0..output_w) |ow| {
                        var sum: f32 = if (bias_data) |bias| bias[oc] else 0.0;

                        for (0..kernel_h) |kh| {
                            const padded_y = oh * stride_h + kh;
                            if (padded_y < pad_top) continue;

                            const input_y = padded_y - pad_top;
                            if (input_y >= input_h) continue;
                            const input_row_base = input_channel_base + input_y * input_w;
                            const weight_row_base = weight_base + kh * kernel_w;

                            for (0..kernel_w) |kw| {
                                const padded_x = ow * stride_w + kw;
                                if (padded_x < pad_left) continue;

                                const input_x = padded_x - pad_left;
                                if (input_x >= input_w) continue;

                                sum += x_data[input_row_base + input_x] *
                                    w_data[weight_row_base + kw];
                            }
                        }

                        output_data[output_row_base + ow] = sum;
                    }
                }
            }
        }
    }
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
    const dtype = try dtypeFromOnnx(value.data_type);

    if (value.data_location) |location| {
        if (location != .DEFAULT) return error.ExternalTensorDataUnsupported;
    }

    if (value.segment != null) return error.TensorSegmentsUnsupported;

    const shape = try shapeFromDims(allocator, value.dims.items);
    errdefer allocator.free(shape);

    const count = try tensor.elementCount(shape);

    return switch (dtype) {
        .float32 => blk: {
            const data = try allocator.alloc(f32, count);
            errdefer allocator.free(data);
            if (value.raw_data) |raw_data| {
                try fillFloat32FromRawData(data, raw_data);
            } else {
                if (value.float_data.items.len != count) return error.TensorElementCountMismatch;
                @memcpy(data, value.float_data.items);
            }
            break :blk .{
                .dtype = .float32,
                .shape = shape,
                .data = .{ .float32 = data },
            };
        },
        .int64 => blk: {
            const data = try allocator.alloc(i64, count);
            errdefer allocator.free(data);
            if (value.raw_data) |raw_data| {
                try fillInt64FromRawData(data, raw_data);
            } else {
                if (value.int64_data.items.len != count) return error.TensorElementCountMismatch;
                @memcpy(data, value.int64_data.items);
            }
            break :blk .{
                .dtype = .int64,
                .shape = shape,
                .data = .{ .int64 = data },
            };
        },
        .int32 => blk: {
            const data = try allocator.alloc(i32, count);
            errdefer allocator.free(data);
            if (value.raw_data) |raw_data| {
                try fillInt32FromRawData(data, raw_data);
            } else {
                if (value.int32_data.items.len != count) return error.TensorElementCountMismatch;
                @memcpy(data, value.int32_data.items);
            }
            break :blk .{
                .dtype = .int32,
                .shape = shape,
                .data = .{ .int32 = data },
            };
        },
        else => return error.UnsupportedTensorDataType,
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

fn dtypeFromOnnx(data_type: ?i32) !tensor.DType {
    const actual = data_type orelse return error.MissingTensorDataType;
    if (actual == @intFromEnum(onnx.TensorProto.DataType.FLOAT)) return .float32;
    if (actual == @intFromEnum(onnx.TensorProto.DataType.INT64)) return .int64;
    if (actual == @intFromEnum(onnx.TensorProto.DataType.INT32)) return .int32;
    if (actual == @intFromEnum(onnx.TensorProto.DataType.UINT8)) return .uint8;
    if (actual == @intFromEnum(onnx.TensorProto.DataType.BOOL)) return .bool;

    return error.UnsupportedTensorDataType;
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

fn fillInt64FromRawData(out: []i64, raw_data: []const u8) !void {
    if (raw_data.len != out.len * @sizeOf(i64)) return error.TensorRawDataLengthMismatch;

    for (out, 0..) |*value, index| {
        const offset = index * @sizeOf(i64);
        const bits =
            @as(u64, raw_data[offset]) |
            (@as(u64, raw_data[offset + 1]) << 8) |
            (@as(u64, raw_data[offset + 2]) << 16) |
            (@as(u64, raw_data[offset + 3]) << 24) |
            (@as(u64, raw_data[offset + 4]) << 32) |
            (@as(u64, raw_data[offset + 5]) << 40) |
            (@as(u64, raw_data[offset + 6]) << 48) |
            (@as(u64, raw_data[offset + 7]) << 56);
        value.* = @bitCast(bits);
    }
}

fn fillInt32FromRawData(out: []i32, raw_data: []const u8) !void {
    if (raw_data.len != out.len * @sizeOf(i32)) return error.TensorRawDataLengthMismatch;

    for (out, 0..) |*value, index| {
        const offset = index * @sizeOf(i32);
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
        0,  1,  2,  3,
        4,  5,  6,  7,
        8,  9,  10, 11,
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
        6.5,  8.5,
        12.5, 14.5,
    };
    const output_data = try output.float32Data();

    try std.testing.expectEqualSlices(usize, &expected_shape, output.shape);
    try std.testing.expectEqualSlices(f32, &expected_data, output_data);
}

test "conv optimizes 1x1 pointwise nchw convolution" {
    const allocator = std.testing.allocator;

    var x = try tensor.Tensor.init(allocator, &.{ 1, 2, 2, 2 }, &.{
        1,  2,
        3,  4,
        10, 20,
        30, 40,
    });
    defer x.deinit(allocator);

    var w = try tensor.Tensor.init(allocator, &.{ 3, 2, 1, 1 }, &.{
        1, 0,
        0, 1,
        2, 3,
    });
    defer w.deinit(allocator);

    var b = try tensor.Tensor.init(allocator, &.{3}, &.{ 0.5, -1, 10 });
    defer b.deinit(allocator);

    var output = try conv(allocator, &x, &w, &b, .{});
    defer output.deinit(allocator);

    try std.testing.expectEqualSlices(usize, &.{ 1, 3, 2, 2 }, output.shape);
    try std.testing.expectEqualSlices(f32, &.{
        1.5, 2.5,
        3.5, 4.5,
        9,   19,
        29,  39,
        42,  74,
        106, 138,
    }, try output.float32Data());
}

test "conv supports grouped depthwise-style convolution" {
    const allocator = std.testing.allocator;

    var x = try tensor.Tensor.init(allocator, &.{ 1, 2, 2, 2 }, &.{
        1,  2,
        3,  4,
        10, 20,
        30, 40,
    });
    defer x.deinit(allocator);

    var w = try tensor.Tensor.init(allocator, &.{ 2, 1, 1, 1 }, &.{ 10, 20 });
    defer w.deinit(allocator);

    var output = try conv(allocator, &x, &w, null, .{ .group = 2 });
    defer output.deinit(allocator);

    try std.testing.expectEqualSlices(usize, &.{ 1, 2, 2, 2 }, output.shape);
    try std.testing.expectEqualSlices(f32, &.{
        10,  20,
        30,  40,
        200, 400,
        600, 800,
    }, try output.float32Data());
}

test "conv optimizes depthwise convolution with padding and stride" {
    const allocator = std.testing.allocator;

    var x = try tensor.Tensor.init(allocator, &.{ 1, 1, 3, 3 }, &.{
        1, 2, 3,
        4, 5, 6,
        7, 8, 9,
    });
    defer x.deinit(allocator);

    var w = try tensor.Tensor.init(allocator, &.{ 1, 1, 3, 3 }, &.{
        1, 1, 1,
        1, 1, 1,
        1, 1, 1,
    });
    defer w.deinit(allocator);

    var b = try tensor.Tensor.init(allocator, &.{1}, &.{0.5});
    defer b.deinit(allocator);

    var output = try conv(allocator, &x, &w, &b, .{
        .pads = .{ 1, 1, 1, 1 },
        .strides = .{ 2, 2 },
        .group = 1,
    });
    defer output.deinit(allocator);

    try std.testing.expectEqualSlices(usize, &.{ 1, 1, 2, 2 }, output.shape);
    try std.testing.expectEqualSlices(f32, &.{
        12.5, 16.5,
        24.5, 28.5,
    }, try output.float32Data());
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

test "elementwise ops support same shape and scalar broadcasting" {
    const allocator = std.testing.allocator;

    var a = try tensor.Tensor.init(allocator, &.{4}, &.{ 2, 4, 6, 8 });
    defer a.deinit(allocator);

    var b = try tensor.Tensor.init(allocator, &.{4}, &.{ 1, 2, 3, 4 });
    defer b.deinit(allocator);

    var scale = try tensor.Tensor.init(allocator, &.{1}, &.{2});
    defer scale.deinit(allocator);

    var sub_output = try sub(allocator, &a, &b);
    defer sub_output.deinit(allocator);
    try std.testing.expectEqualSlices(f32, &.{ 1, 2, 3, 4 }, try sub_output.float32Data());

    var mul_output = try mul(allocator, &a, &scale);
    defer mul_output.deinit(allocator);
    try std.testing.expectEqualSlices(f32, &.{ 4, 8, 12, 16 }, try mul_output.float32Data());

    var div_output = try div(allocator, &a, &scale);
    defer div_output.deinit(allocator);
    try std.testing.expectEqualSlices(f32, &.{ 1, 2, 3, 4 }, try div_output.float32Data());
}

test "activation ops transform float tensors" {
    const allocator = std.testing.allocator;

    var input = try tensor.Tensor.init(allocator, &.{3}, &.{ -1, 0, 1 });
    defer input.deinit(allocator);

    var relu_output = try relu(allocator, &input);
    defer relu_output.deinit(allocator);
    try std.testing.expectEqualSlices(f32, &.{ 0, 0, 1 }, try relu_output.float32Data());

    var sigmoid_output = try sigmoid(allocator, &input);
    defer sigmoid_output.deinit(allocator);
    const sigmoid_data = try sigmoid_output.float32Data();
    try std.testing.expectApproxEqAbs(@as(f32, 0.26894143), sigmoid_data[0], 0.00001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), sigmoid_data[1], 0.00001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.7310586), sigmoid_data[2], 0.00001);

    var tanh_output = try tanh(allocator, &input);
    defer tanh_output.deinit(allocator);
    const tanh_data = try tanh_output.float32Data();
    try std.testing.expectApproxEqAbs(@as(f32, -0.7615942), tanh_data[0], 0.00001);
    try std.testing.expectApproxEqAbs(@as(f32, 0), tanh_data[1], 0.00001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.7615942), tanh_data[2], 0.00001);
}

test "reshape infers one target dimension" {
    const allocator = std.testing.allocator;

    var input = try tensor.Tensor.init(allocator, &.{ 2, 3, 4 }, &.{
        0,  1,  2,  3,
        4,  5,  6,  7,
        8,  9,  10, 11,
        12, 13, 14, 15,
        16, 17, 18, 19,
        20, 21, 22, 23,
    });
    defer input.deinit(allocator);

    var shape = try tensor.Tensor.initInt64(allocator, &.{2}, &.{ 6, -1 });
    defer shape.deinit(allocator);

    var output = try reshape(allocator, &input, &shape, false);
    defer output.deinit(allocator);

    try std.testing.expectEqualSlices(usize, &.{ 6, 4 }, output.shape);
    try std.testing.expectEqualSlices(f32, try input.float32Data(), try output.float32Data());
}

test "transpose permutes a ranked float tensor" {
    const allocator = std.testing.allocator;

    var input = try tensor.Tensor.init(allocator, &.{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer input.deinit(allocator);

    var output = try transpose(allocator, &input, &.{ 1, 0 });
    defer output.deinit(allocator);

    try std.testing.expectEqualSlices(usize, &.{ 3, 2 }, output.shape);
    try std.testing.expectEqualSlices(f32, &.{ 1, 4, 2, 5, 3, 6 }, try output.float32Data());
}

test "clip clamps float tensor with scalar bounds" {
    const allocator = std.testing.allocator;

    var input = try tensor.Tensor.init(allocator, &.{5}, &.{ -2, -1, 0, 1, 2 });
    defer input.deinit(allocator);

    var min_value = try tensor.Tensor.init(allocator, &.{}, &.{-0.5});
    defer min_value.deinit(allocator);

    var max_value = try tensor.Tensor.init(allocator, &.{}, &.{1.5});
    defer max_value.deinit(allocator);

    var output = try clip(allocator, &input, &min_value, &max_value, null, null);
    defer output.deinit(allocator);

    try std.testing.expectEqualSlices(usize, &.{5}, output.shape);
    try std.testing.expectEqualSlices(f32, &.{ -0.5, -0.5, 0, 1, 1.5 }, try output.float32Data());
}

test "globalAveragePool reduces spatial dimensions" {
    const allocator = std.testing.allocator;

    var input = try tensor.Tensor.init(allocator, &.{ 1, 2, 2, 2 }, &.{
        1,  2,
        3,  4,
        10, 20,
        30, 40,
    });
    defer input.deinit(allocator);

    var output = try globalAveragePool(allocator, &input);
    defer output.deinit(allocator);

    try std.testing.expectEqualSlices(usize, &.{ 1, 2, 1, 1 }, output.shape);
    try std.testing.expectEqualSlices(f32, &.{ 2.5, 25 }, try output.float32Data());
}

test "shapeTensor returns int64 tensor dimensions" {
    const allocator = std.testing.allocator;

    var input = try tensor.Tensor.init(allocator, &.{ 1, 3, 4, 4 }, &([_]f32{0} ** (1 * 3 * 4 * 4)));
    defer input.deinit(allocator);

    var output = try shapeTensor(allocator, &input);
    defer output.deinit(allocator);

    try std.testing.expectEqualSlices(usize, &.{4}, output.shape);
    switch (output.data) {
        .int64 => |values| try std.testing.expectEqualSlices(i64, &.{ 1, 3, 4, 4 }, values),
        else => return error.ExpectedInt64Tensor,
    }
}

test "gather selects from int64 vector with integer indices" {
    const allocator = std.testing.allocator;

    var data = try tensor.Tensor.initInt64(allocator, &.{4}, &.{ 1, 3, 224, 224 });
    defer data.deinit(allocator);

    var indices = try tensor.Tensor.initInt64(allocator, &.{2}, &.{ 0, -1 });
    defer indices.deinit(allocator);

    var output = try gather(allocator, &data, &indices, 0);
    defer output.deinit(allocator);

    try std.testing.expectEqualSlices(usize, &.{2}, output.shape);
    switch (output.data) {
        .int64 => |values| try std.testing.expectEqualSlices(i64, &.{ 1, 224 }, values),
        else => return error.ExpectedInt64Tensor,
    }
}

test "unsqueeze inserts singleton axes" {
    const allocator = std.testing.allocator;

    var input = try tensor.Tensor.initInt64(allocator, &.{2}, &.{ 224, 224 });
    defer input.deinit(allocator);

    var output = try unsqueeze(allocator, &input, &.{ 0, 2 });
    defer output.deinit(allocator);

    try std.testing.expectEqualSlices(usize, &.{ 1, 2, 1 }, output.shape);
    switch (output.data) {
        .int64 => |values| try std.testing.expectEqualSlices(i64, &.{ 224, 224 }, values),
        else => return error.ExpectedInt64Tensor,
    }
}

test "concat joins int64 tensors along axis zero" {
    const allocator = std.testing.allocator;

    var a = try tensor.Tensor.initInt64(allocator, &.{2}, &.{ 1, 3 });
    defer a.deinit(allocator);

    var b = try tensor.Tensor.initInt64(allocator, &.{2}, &.{ 224, 224 });
    defer b.deinit(allocator);

    var output = try concat(allocator, &.{ &a, &b }, 0);
    defer output.deinit(allocator);

    try std.testing.expectEqualSlices(usize, &.{4}, output.shape);
    switch (output.data) {
        .int64 => |values| try std.testing.expectEqualSlices(i64, &.{ 1, 3, 224, 224 }, values),
        else => return error.ExpectedInt64Tensor,
    }
}

test "maxPool reduces 2d nchw windows" {
    const allocator = std.testing.allocator;

    var input = try tensor.Tensor.init(allocator, &.{ 1, 1, 4, 4 }, &.{
        1,  2,  3,  4,
        5,  6,  7,  8,
        9,  10, 11, 12,
        13, 14, 15, 16,
    });
    defer input.deinit(allocator);

    var output = try maxPool(allocator, &input, .{
        .kernel_shape = .{ 2, 2 },
        .strides = .{ 2, 2 },
    });
    defer output.deinit(allocator);

    try std.testing.expectEqualSlices(usize, &.{ 1, 1, 2, 2 }, output.shape);
    try std.testing.expectEqualSlices(f32, &.{ 6, 8, 14, 16 }, try output.float32Data());
}

test "averagePool reduces valid 2d nchw windows" {
    const allocator = std.testing.allocator;

    var input = try tensor.Tensor.init(allocator, &.{ 1, 1, 2, 2 }, &.{
        1, 2,
        3, 4,
    });
    defer input.deinit(allocator);

    var output = try averagePool(allocator, &input, .{
        .kernel_shape = .{ 2, 2 },
    });
    defer output.deinit(allocator);

    try std.testing.expectEqualSlices(usize, &.{ 1, 1, 1, 1 }, output.shape);
    try std.testing.expectEqualSlices(f32, &.{2.5}, try output.float32Data());
}

test "batchNormalization applies per-channel inference formula" {
    const allocator = std.testing.allocator;

    var input = try tensor.Tensor.init(allocator, &.{ 1, 2, 1, 2 }, &.{ 1, 3, 10, 14 });
    defer input.deinit(allocator);

    var scale = try tensor.Tensor.init(allocator, &.{2}, &.{ 2, 0.5 });
    defer scale.deinit(allocator);

    var bias = try tensor.Tensor.init(allocator, &.{2}, &.{ 1, -1 });
    defer bias.deinit(allocator);

    var mean = try tensor.Tensor.init(allocator, &.{2}, &.{ 2, 10 });
    defer mean.deinit(allocator);

    var variance = try tensor.Tensor.init(allocator, &.{2}, &.{ 1, 4 });
    defer variance.deinit(allocator);

    var output = try batchNormalization(allocator, &input, &scale, &bias, &mean, &variance, .{ .epsilon = 0 });
    defer output.deinit(allocator);

    try std.testing.expectEqualSlices(usize, &.{ 1, 2, 1, 2 }, output.shape);
    try std.testing.expectEqualSlices(f32, &.{ -1, 3, -1, 0 }, try output.float32Data());
}

test "squeeze removes selected singleton axes" {
    const allocator = std.testing.allocator;

    var input = try tensor.Tensor.initInt64(allocator, &.{ 1, 2, 1 }, &.{ 5, 6 });
    defer input.deinit(allocator);

    var output = try squeeze(allocator, &input, &.{ 0, 2 });
    defer output.deinit(allocator);

    try std.testing.expectEqualSlices(usize, &.{2}, output.shape);
    switch (output.data) {
        .int64 => |values| try std.testing.expectEqualSlices(i64, &.{ 5, 6 }, values),
        else => return error.ExpectedInt64Tensor,
    }
}

test "cast converts existing tensor dtypes" {
    const allocator = std.testing.allocator;

    var input = try tensor.Tensor.initInt64(allocator, &.{3}, &.{ 0, 2, 4 });
    defer input.deinit(allocator);

    var output = try cast(allocator, &input, .float32);
    defer output.deinit(allocator);

    try std.testing.expectEqual(tensor.DType.float32, output.dtype);
    try std.testing.expectEqualSlices(f32, &.{ 0, 2, 4 }, try output.float32Data());
}

test "slice extracts positive-step tensor ranges" {
    const allocator = std.testing.allocator;

    var input = try tensor.Tensor.init(allocator, &.{ 2, 3, 4 }, &.{
        0,  1,  2,  3,
        4,  5,  6,  7,
        8,  9,  10, 11,
        12, 13, 14, 15,
        16, 17, 18, 19,
        20, 21, 22, 23,
    });
    defer input.deinit(allocator);

    var starts = try tensor.Tensor.initInt64(allocator, &.{2}, &.{ 0, 1 });
    defer starts.deinit(allocator);

    var ends = try tensor.Tensor.initInt64(allocator, &.{2}, &.{ 2, 3 });
    defer ends.deinit(allocator);

    var axes = try tensor.Tensor.initInt64(allocator, &.{2}, &.{ 0, 1 });
    defer axes.deinit(allocator);

    var steps = try tensor.Tensor.initInt64(allocator, &.{2}, &.{ 1, 1 });
    defer steps.deinit(allocator);

    var output = try slice(allocator, &input, &starts, &ends, &axes, &steps);
    defer output.deinit(allocator);

    try std.testing.expectEqualSlices(usize, &.{ 2, 2, 4 }, output.shape);
    try std.testing.expectEqualSlices(f32, &.{
        4,  5,  6,  7,
        8,  9,  10, 11,
        16, 17, 18, 19,
        20, 21, 22, 23,
    }, try output.float32Data());
}
