const std = @import("std");
const tensor = @import("tensor.zig");

pub const GemmOptions = struct {
    alpha: f32 = 1.0,
    beta: f32 = 1.0,
    trans_a: bool = false,
    trans_b: bool = false,
};

pub fn flatten(allocator: std.mem.Allocator, input: *const tensor.Tensor, axis: i64) !tensor.Tensor {
    const normalized_axis = try normalizeAxis(axis, input.shape.len, true);
    const outer = try tensor.elementCount(input.shape[0..normalized_axis]);
    const inner = try tensor.elementCount(input.shape[normalized_axis..]);

    const shape = [_]usize{ outer, inner };
    return tensor.Tensor.init(allocator, &shape, input.data);
}

pub fn gemm(
    allocator: std.mem.Allocator,
    a: *const tensor.Tensor,
    b: *const tensor.Tensor,
    c: ?*const tensor.Tensor,
    options: GemmOptions,
) !tensor.Tensor {
    if (a.shape.len != 2 or b.shape.len != 2) return error.GemmRequiresMatrices;

    const a_rows = if (options.trans_a) a.shape[1] else a.shape[0];
    const a_cols = if (options.trans_a) a.shape[0] else a.shape[1];
    const b_rows = if (options.trans_b) b.shape[1] else b.shape[0];
    const b_cols = if (options.trans_b) b.shape[0] else b.shape[1];

    if (a_cols != b_rows) return error.GemmDimensionMismatch;

    const output_shape = [_]usize{ a_rows, b_cols };
    var output = try tensor.Tensor.initZeros(allocator, &output_shape);
    errdefer output.deinit(allocator);

    for (0..a_rows) |row| {
        for (0..b_cols) |col| {
            var sum: f32 = 0.0;

            for (0..a_cols) |k| {
                sum += matrixValue(a, row, k, options.trans_a) *
                    matrixValue(b, k, col, options.trans_b);
            }

            const index = row * b_cols + col;
            output.data[index] = options.alpha * sum;

            if (c) |bias| {
                output.data[index] += options.beta * try biasValue(bias, row, col, a_rows, b_cols);
            }
        }
    }

    return output;
}

pub fn softmax(allocator: std.mem.Allocator, input: *const tensor.Tensor, axis: i64) !tensor.Tensor {
    if (input.shape.len == 0) return error.SoftmaxRequiresRankedTensor;

    const normalized_axis = try normalizeAxis(axis, input.shape.len, false);
    const outer = try tensor.elementCount(input.shape[0..normalized_axis]);
    const axis_len = input.shape[normalized_axis];
    const inner = try tensor.elementCount(input.shape[normalized_axis + 1 ..]);

    var output = try tensor.Tensor.initZeros(allocator, input.shape);
    errdefer output.deinit(allocator);

    for (0..outer) |outer_index| {
        for (0..inner) |inner_index| {
            const base = outer_index * axis_len * inner + inner_index;
            var max_value = input.data[base];

            for (1..axis_len) |axis_index| {
                const value = input.data[base + axis_index * inner];
                if (value > max_value) max_value = value;
            }

            var sum: f32 = 0.0;

            for (0..axis_len) |axis_index| {
                const index = base + axis_index * inner;
                const value = @exp(input.data[index] - max_value);
                output.data[index] = value;
                sum += value;
            }

            for (0..axis_len) |axis_index| {
                const index = base + axis_index * inner;
                output.data[index] /= sum;
            }
        }
    }

    return output;
}

fn normalizeAxis(axis: i64, rank: usize, allow_end: bool) !usize {
    const rank_i64 = std.math.cast(i64, rank) orelse return error.RankTooLarge;
    const normalized = if (axis < 0) axis + rank_i64 else axis;
    const max_axis = if (allow_end) rank_i64 else rank_i64 - 1;

    if (normalized < 0 or normalized > max_axis) return error.AxisOutOfBounds;

    return std.math.cast(usize, normalized) orelse error.AxisOutOfBounds;
}

fn matrixValue(input: *const tensor.Tensor, row: usize, col: usize, transposed: bool) f32 {
    const source_row = if (transposed) col else row;
    const source_col = if (transposed) row else col;
    return input.data[source_row * input.shape[1] + source_col];
}

fn biasValue(
    bias: *const tensor.Tensor,
    row: usize,
    col: usize,
    output_rows: usize,
    output_cols: usize,
) !f32 {
    if (bias.shape.len == 0) return bias.data[0];

    if (bias.shape.len == 1) {
        if (bias.shape[0] == 1) return bias.data[0];
        if (bias.shape[0] == output_cols) return bias.data[col];
        if (bias.shape[0] == output_rows) return bias.data[row];
        return error.GemmBiasShapeMismatch;
    }

    if (bias.shape.len == 2) {
        const rows = bias.shape[0];
        const cols = bias.shape[1];

        if (rows == output_rows and cols == output_cols) return bias.data[row * cols + col];
        if (rows == 1 and cols == output_cols) return bias.data[col];
        if (rows == output_rows and cols == 1) return bias.data[row];
        if (rows == 1 and cols == 1) return bias.data[0];
    }

    return error.GemmBiasShapeMismatch;
}
