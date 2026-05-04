const std = @import("std");

pub const DType = enum {
    float32,
};

pub const Tensor = struct {
    dtype: DType = .float32,
    shape: []usize,
    data: []f32,

    pub fn init(
        allocator: std.mem.Allocator,
        shape: []const usize,
        data: []const f32,
    ) !Tensor {
        const expected = try elementCount(shape);

        if (expected != data.len) {
            return error.TensorElementCountMismatch;
        }

        const owned_shape = try allocator.dupe(usize, shape);
        errdefer allocator.free(owned_shape);

        const owned_data = try allocator.dupe(f32, data);
        errdefer allocator.free(owned_data);

        return .{
            .shape = owned_shape,
            .data = owned_data,
        };
    }

    pub fn initZeros(
        allocator: std.mem.Allocator,
        shape: []const usize,
    ) !Tensor {
        const count = try elementCount(shape);

        const owned_shape = try allocator.dupe(usize, shape);
        errdefer allocator.free(owned_shape);

        const owned_data = try allocator.alloc(f32, count);
        @memset(owned_data, 0);

        return .{
            .shape = owned_shape,
            .data = owned_data,
        };
    }

    pub fn deinit(self: *Tensor, allocator: std.mem.Allocator) void {
        allocator.free(self.shape);
        allocator.free(self.data);

        self.* = undefined;
    }

    pub fn elementCountSelf(self: Tensor) usize {
        return self.data.len;
    }
};

pub fn elementCount(shape: []const usize) !usize {
    var count: usize = 1;

    for (shape) |dim| {
        if (dim == 0) {
            return error.ZeroDimensionUnsupported;
        }

        count = try std.math.mul(usize, count, dim);
    }

    return count;
}
