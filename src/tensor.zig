const std = @import("std");

pub const DType = enum {
    float32,
    int64,
    int32,
    uint8,
    bool,
};

pub const TensorData = union(DType) {
    float32: []f32,
    int64: []i64,
    int32: []i32,
    uint8: []u8,
    bool: []bool,
};

pub const Tensor = struct {
    dtype: DType,
    shape: []usize,
    data: TensorData,

    pub fn init(
        allocator: std.mem.Allocator,
        shape: []const usize,
        data: []const f32,
    ) !Tensor {
        return initFloat32(allocator, shape, data);
    }

    pub fn initZeros(
        allocator: std.mem.Allocator,
        shape: []const usize,
    ) !Tensor {
        return initZerosFloat32(allocator, shape);
    }

    pub fn initFloat32(
        allocator: std.mem.Allocator,
        shape: []const usize,
        data: []const f32,
    ) !Tensor {
        try validateElementCount(shape, data.len);

        const owned_shape = try allocator.dupe(usize, shape);
        errdefer allocator.free(owned_shape);

        const owned_data = try allocator.dupe(f32, data);
        errdefer allocator.free(owned_data);

        return .{
            .dtype = .float32,
            .shape = owned_shape,
            .data = .{ .float32 = owned_data },
        };
    }

    pub fn initInt64(
        allocator: std.mem.Allocator,
        shape: []const usize,
        data: []const i64,
    ) !Tensor {
        try validateElementCount(shape, data.len);

        const owned_shape = try allocator.dupe(usize, shape);
        errdefer allocator.free(owned_shape);

        const owned_data = try allocator.dupe(i64, data);
        errdefer allocator.free(owned_data);

        return .{
            .dtype = .int64,
            .shape = owned_shape,
            .data = .{ .int64 = owned_data },
        };
    }

    pub fn initInt32(
        allocator: std.mem.Allocator,
        shape: []const usize,
        data: []const i32,
    ) !Tensor {
        try validateElementCount(shape, data.len);

        const owned_shape = try allocator.dupe(usize, shape);
        errdefer allocator.free(owned_shape);

        const owned_data = try allocator.dupe(i32, data);
        errdefer allocator.free(owned_data);

        return .{
            .dtype = .int32,
            .shape = owned_shape,
            .data = .{ .int32 = owned_data },
        };
    }

    pub fn initUint8(
        allocator: std.mem.Allocator,
        shape: []const usize,
        data: []const u8,
    ) !Tensor {
        try validateElementCount(shape, data.len);

        const owned_shape = try allocator.dupe(usize, shape);
        errdefer allocator.free(owned_shape);

        const owned_data = try allocator.dupe(u8, data);
        errdefer allocator.free(owned_data);

        return .{
            .dtype = .uint8,
            .shape = owned_shape,
            .data = .{ .uint8 = owned_data },
        };
    }

    pub fn initBool(
        allocator: std.mem.Allocator,
        shape: []const usize,
        data: []const bool,
    ) !Tensor {
        try validateElementCount(shape, data.len);

        const owned_shape = try allocator.dupe(usize, shape);
        errdefer allocator.free(owned_shape);

        const owned_data = try allocator.dupe(bool, data);
        errdefer allocator.free(owned_data);

        return .{
            .dtype = .bool,
            .shape = owned_shape,
            .data = .{ .bool = owned_data },
        };
    }

    pub fn initZerosFloat32(
        allocator: std.mem.Allocator,
        shape: []const usize,
    ) !Tensor {
        const count = try elementCount(shape);

        const owned_shape = try allocator.dupe(usize, shape);
        errdefer allocator.free(owned_shape);

        const owned_data = try allocator.alloc(f32, count);
        @memset(owned_data, 0);

        return .{
            .dtype = .float32,
            .shape = owned_shape,
            .data = .{ .float32 = owned_data },
        };
    }

    pub fn deinit(self: *Tensor, allocator: std.mem.Allocator) void {
        allocator.free(self.shape);

        switch (self.data) {
            .float32 => |data| allocator.free(data),
            .int64 => |data| allocator.free(data),
            .int32 => |data| allocator.free(data),
            .uint8 => |data| allocator.free(data),
            .bool => |data| allocator.free(data),
        }

        self.* = undefined;
    }

    pub fn elementCountSelf(self: Tensor) usize {
        return switch (self.data) {
            .float32 => |data| data.len,
            .int64 => |data| data.len,
            .int32 => |data| data.len,
            .uint8 => |data| data.len,
            .bool => |data| data.len,
        };
    }

    pub fn float32Data(self: *const Tensor) ![]const f32 {
        return switch (self.data) {
            .float32 => |data| data,
            else => error.ExpectedFloat32Tensor,
        };
    }

    pub fn float32DataMut(self: *Tensor) ![]f32 {
        return switch (self.data) {
            .float32 => |data| data,
            else => error.ExpectedFloat32Tensor,
        };
    }
};

fn validateElementCount(shape: []const usize, actual: usize) !void {
    const expected = try elementCount(shape);

    if (expected != actual) {
        return error.TensorElementCountMismatch;
    }
}

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

test "tensor stores non-float dtypes" {
    const allocator = std.testing.allocator;

    var int_tensor = try Tensor.initInt64(allocator, &.{ 2, 2 }, &.{ 1, 2, 3, 4 });
    defer int_tensor.deinit(allocator);

    try std.testing.expectEqual(DType.int64, int_tensor.dtype);
    try std.testing.expectEqualSlices(usize, &.{ 2, 2 }, int_tensor.shape);
    try std.testing.expectEqual(@as(usize, 4), int_tensor.elementCountSelf());

    switch (int_tensor.data) {
        .int64 => |values| try std.testing.expectEqualSlices(i64, &.{ 1, 2, 3, 4 }, values),
        else => return error.ExpectedInt64Tensor,
    }

    try std.testing.expectError(error.ExpectedFloat32Tensor, int_tensor.float32Data());

    var bool_tensor = try Tensor.initBool(allocator, &.{3}, &.{ true, false, true });
    defer bool_tensor.deinit(allocator);

    try std.testing.expectEqual(DType.bool, bool_tensor.dtype);

    switch (bool_tensor.data) {
        .bool => |values| try std.testing.expectEqualSlices(bool, &.{ true, false, true }, values),
        else => return error.ExpectedBoolTensor,
    }
}
