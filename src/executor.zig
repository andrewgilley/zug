const std = @import("std");
const onnx = @import("proto/onnx.pb.zig");
const tensor = @import("tensor.zig");
const ops = @import("ops.zig");

const max_tensor_file_bytes = 100 * 1024 * 1024;

pub const InputSpec = struct {
    name: []const u8,
    path: []const u8,
};

pub const Output = struct {
    name: []const u8,
    value: *const tensor.Tensor,
};

pub const Executor = struct {
    allocator: std.mem.Allocator,
    graph: *const onnx.GraphProto,
    values: std.StringHashMap(tensor.Tensor),
    outputs: std.ArrayList(Output) = .empty,

    pub fn init(allocator: std.mem.Allocator, model: *const onnx.ModelProto) !Executor {
        const graph = if (model.graph) |*graph| graph else return error.MissingGraph;

        var self = Executor{
            .allocator = allocator,
            .graph = graph,
            .values = std.StringHashMap(tensor.Tensor).init(allocator),
        };
        errdefer self.deinit();

        for (graph.initializer.items) |*initializer| {
            const name = initializer.name orelse return error.MissingInitializerName;
            const value = try tensorFromInitializer(allocator, initializer);
            try self.put(name, value);
        }

        return self;
    }

    pub fn deinit(self: *Executor) void {
        var values = self.values.valueIterator();
        while (values.next()) |value| {
            value.deinit(self.allocator);
        }

        self.values.deinit();
        self.outputs.deinit(self.allocator);
    }

    pub fn loadInputsFromFiles(self: *Executor, inputs: anytype) !void {
        for (inputs) |input| {
            if (self.isInitializerName(input.name)) return error.InputOverridesInitializer;
            if (self.values.contains(input.name)) return error.InputAlreadyBound;

            const info = try self.graphInput(input.name);
            const value = try tensorFromRawFile(self.allocator, info, input.path);
            try self.put(input.name, value);
        }

        try self.ensureRequiredInputsBound();
    }

    pub fn setInput(self: *Executor, name: []const u8, value: tensor.Tensor) !void {
        if (self.isInitializerName(name)) return error.InputOverridesInitializer;
        _ = try self.graphInput(name);

        try self.put(name, value);
    }

    pub fn execute(self: *Executor) ![]const Output {
        try self.ensureRequiredInputsBound();

        for (self.graph.node.items) |*node| {
            try self.runNode(node);
        }

        self.outputs.clearRetainingCapacity();

        for (self.graph.output.items) |*output_info| {
            const name = output_info.name orelse return error.MissingOutputName;
            const value = self.values.getPtr(name) orelse return error.MissingGraphOutput;
            try self.outputs.append(self.allocator, .{
                .name = name,
                .value = value,
            });
        }

        return self.outputs.items;
    }

    fn runNode(self: *Executor, node: *const onnx.NodeProto) !void {
        try ensureDefaultDomain(node);

        const op_type = node.op_type orelse return error.MissingOpType;

        if (node.output.items.len != 1 or node.output.items[0].len == 0) {
            return error.InvalidNodeOutput;
        }

        const output_name = node.output.items[0];

        const result = if (std.mem.eql(u8, op_type, "Add")) blk: {
            const a = try self.requiredNodeInput(node, 0);
            const b = try self.requiredNodeInput(node, 1);
            break :blk try ops.add(self.allocator, a, b);
        } else if (std.mem.eql(u8, op_type, "Constant")) blk: {
            break :blk try ops.constant(self.allocator, node);
        } else if (std.mem.eql(u8, op_type, "Conv")) blk: {
            const x = try self.requiredNodeInput(node, 0);
            const w = try self.requiredNodeInput(node, 1);
            const b = try self.optionalNodeInput(node, 2);

            break :blk try ops.conv(self.allocator, x, w, b, .{
                .pads = try padsAttr(node),
                .strides = try pairAttr(node, "strides", .{ 1, 1 }),
                .dilations = try pairAttr(node, "dilations", .{ 1, 1 }),
                .group = std.math.cast(usize, try intAttr(node, "group", 1)) orelse return error.InvalidConvGroup,
            });
        } else if (std.mem.eql(u8, op_type, "Flatten")) blk: {
            const input = try self.requiredNodeInput(node, 0);
            const axis = try intAttr(node, "axis", 1);
            break :blk try ops.flatten(self.allocator, input, axis);
        } else if (std.mem.eql(u8, op_type, "Gemm")) blk: {
            const a = try self.requiredNodeInput(node, 0);
            const b = try self.requiredNodeInput(node, 1);
            const c = try self.optionalNodeInput(node, 2);

            break :blk try ops.gemm(self.allocator, a, b, c, .{
                .alpha = try floatAttr(node, "alpha", 1.0),
                .beta = try floatAttr(node, "beta", 1.0),
                .trans_a = (try intAttr(node, "transA", 0)) != 0,
                .trans_b = (try intAttr(node, "transB", 0)) != 0,
            });
        } else if (std.mem.eql(u8, op_type, "MatMul")) blk: {
            const a = try self.requiredNodeInput(node, 0);
            const b = try self.requiredNodeInput(node, 1);
            break :blk try ops.matmul(self.allocator, a, b);
        } else if (std.mem.eql(u8, op_type, "Softmax")) blk: {
            const input = try self.requiredNodeInput(node, 0);
            const axis = try intAttr(node, "axis", -1);
            break :blk try ops.softmax(self.allocator, input, axis);
        } else {
            return error.UnsupportedOperator;
        };

        try self.put(output_name, result);
    }

    fn put(self: *Executor, name: []const u8, value: tensor.Tensor) !void {
        var owned = value;
        errdefer owned.deinit(self.allocator);

        if (self.values.getPtr(name)) |existing| {
            existing.deinit(self.allocator);
            existing.* = owned;
            return;
        }

        try self.values.put(name, owned);
    }

    fn requiredNodeInput(self: *Executor, node: *const onnx.NodeProto, index: usize) !*const tensor.Tensor {
        if (node.input.items.len <= index or node.input.items[index].len == 0) {
            return error.MissingNodeInput;
        }

        if (self.values.getPtr(node.input.items[index])) |value| {
            return value;
        }

        return error.MissingNodeInputValue;
    }

    fn optionalNodeInput(self: *Executor, node: *const onnx.NodeProto, index: usize) !?*const tensor.Tensor {
        if (node.input.items.len <= index or node.input.items[index].len == 0) return null;

        if (self.values.getPtr(node.input.items[index])) |value| {
            return value;
        }

        return error.MissingNodeInputValue;
    }

    fn graphInput(self: *const Executor, name: []const u8) !*const onnx.ValueInfoProto {
        for (self.graph.input.items) |*input| {
            if (input.name) |input_name| {
                if (std.mem.eql(u8, input_name, name)) return input;
            }
        }

        return error.UnknownGraphInput;
    }

    fn ensureRequiredInputsBound(self: *const Executor) !void {
        for (self.graph.input.items) |*input| {
            const name = input.name orelse return error.MissingInputName;
            if (self.isInitializerName(name)) continue;
            if (!self.values.contains(name)) return error.MissingGraphInput;
        }
    }

    fn isInitializerName(self: *const Executor, name: []const u8) bool {
        for (self.graph.initializer.items) |*initializer| {
            if (initializer.name) |initializer_name| {
                if (std.mem.eql(u8, initializer_name, name)) return true;
            }
        }

        return false;
    }
};

fn tensorFromInitializer(allocator: std.mem.Allocator, initializer: *const onnx.TensorProto) !tensor.Tensor {
    const dtype = try dtypeFromOnnx(initializer.data_type);

    if (initializer.data_location) |location| {
        if (location != .DEFAULT) return error.ExternalTensorDataUnsupported;
    }

    if (initializer.segment != null) return error.TensorSegmentsUnsupported;

    const shape = try shapeFromDims(allocator, initializer.dims.items);
    defer allocator.free(shape);

    const count = try tensor.elementCount(shape);

    if (initializer.raw_data) |raw_data| {
        return tensorFromRawData(allocator, dtype, shape, raw_data);
    }

    return switch (dtype) {
        .float32 => tensor.Tensor.initFloat32(allocator, shape, try checkedItems(f32, initializer.float_data.items, count)),
        .int64 => tensor.Tensor.initInt64(allocator, shape, try checkedItems(i64, initializer.int64_data.items, count)),
        .int32 => tensor.Tensor.initInt32(allocator, shape, try checkedItems(i32, initializer.int32_data.items, count)),
        .uint8 => blk: {
            const data = try allocator.alloc(u8, count);
            defer allocator.free(data);
            try fillUint8FromInt32Data(data, initializer.int32_data.items);
            break :blk tensor.Tensor.initUint8(allocator, shape, data);
        },
        .bool => blk: {
            const data = try allocator.alloc(bool, count);
            defer allocator.free(data);
            try fillBoolFromInt32Data(data, initializer.int32_data.items);
            break :blk tensor.Tensor.initBool(allocator, shape, data);
        },
    };
}

fn tensorFromRawFile(
    allocator: std.mem.Allocator,
    input: *const onnx.ValueInfoProto,
    path: []const u8,
) !tensor.Tensor {
    const dtype = try dtypeFromValueInfo(input);
    const shape = try shapeFromValueInfo(allocator, input);
    defer allocator.free(shape);

    const count = try tensor.elementCount(shape);
    const expected_bytes = try std.math.mul(usize, count, elementByteSize(dtype));

    const bytes = try std.Io.Dir.cwd().readFileAlloc(
        std.Options.debug_io,
        path,
        allocator,
        .limited(max_tensor_file_bytes),
    );
    defer allocator.free(bytes);

    if (bytes.len != expected_bytes) return error.InputByteLengthMismatch;

    return tensorFromRawData(allocator, dtype, shape, bytes);
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

fn shapeFromValueInfo(allocator: std.mem.Allocator, input: *const onnx.ValueInfoProto) ![]usize {
    const type_proto = input.type orelse return error.MissingInputType;
    const type_value = type_proto.value orelse return error.MissingInputType;

    const tensor_type = switch (type_value) {
        .tensor_type => |value| value,
        else => return error.UnsupportedInputType,
    };

    const shape_proto = tensor_type.shape orelse return error.MissingInputShape;
    const shape = try allocator.alloc(usize, shape_proto.dim.items.len);
    errdefer allocator.free(shape);

    for (shape_proto.dim.items, 0..) |dim, index| {
        const value = dim.value orelse return error.DynamicDimensionUnsupported;
        shape[index] = switch (value) {
            .dim_value => |dim_value| blk: {
                if (dim_value <= 0) return error.InvalidTensorDimension;
                break :blk std.math.cast(usize, dim_value) orelse return error.DimensionTooLarge;
            },
            .dim_param => return error.DynamicDimensionUnsupported,
        };
    }

    return shape;
}

fn dtypeFromValueInfo(input: *const onnx.ValueInfoProto) !tensor.DType {
    const type_proto = input.type orelse return error.MissingInputType;
    const type_value = type_proto.value orelse return error.MissingInputType;

    const tensor_type = switch (type_value) {
        .tensor_type => |value| value,
        else => return error.UnsupportedInputType,
    };

    return dtypeFromOnnx(tensor_type.elem_type);
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

fn elementByteSize(dtype: tensor.DType) usize {
    return switch (dtype) {
        .float32 => @sizeOf(f32),
        .int64 => @sizeOf(i64),
        .int32 => @sizeOf(i32),
        .uint8 => @sizeOf(u8),
        .bool => @sizeOf(u8),
    };
}

fn tensorFromRawData(
    allocator: std.mem.Allocator,
    dtype: tensor.DType,
    shape: []const usize,
    raw_data: []const u8,
) !tensor.Tensor {
    const count = try tensor.elementCount(shape);
    const expected_bytes = try std.math.mul(usize, count, elementByteSize(dtype));
    if (raw_data.len != expected_bytes) return error.TensorRawDataLengthMismatch;

    return switch (dtype) {
        .float32 => blk: {
            const data = try allocator.alloc(f32, count);
            defer allocator.free(data);
            try fillFloat32FromRawData(data, raw_data);
            break :blk tensor.Tensor.initFloat32(allocator, shape, data);
        },
        .int64 => blk: {
            const data = try allocator.alloc(i64, count);
            defer allocator.free(data);
            try fillInt64FromRawData(data, raw_data);
            break :blk tensor.Tensor.initInt64(allocator, shape, data);
        },
        .int32 => blk: {
            const data = try allocator.alloc(i32, count);
            defer allocator.free(data);
            try fillInt32FromRawData(data, raw_data);
            break :blk tensor.Tensor.initInt32(allocator, shape, data);
        },
        .uint8 => tensor.Tensor.initUint8(allocator, shape, raw_data),
        .bool => blk: {
            const data = try allocator.alloc(bool, count);
            defer allocator.free(data);
            for (data, raw_data) |*value, raw| {
                value.* = raw != 0;
            }
            break :blk tensor.Tensor.initBool(allocator, shape, data);
        },
    };
}

fn checkedItems(comptime T: type, items: []const T, expected: usize) ![]const T {
    if (items.len != expected) return error.TensorElementCountMismatch;
    return items;
}

fn fillFloat32FromRawData(out: []f32, raw_data: []const u8) !void {
    if (raw_data.len != out.len * @sizeOf(f32)) return error.TensorRawDataLengthMismatch;

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

fn fillUint8FromInt32Data(out: []u8, items: []const i32) !void {
    if (items.len != out.len) return error.TensorElementCountMismatch;

    for (out, items) |*value, item| {
        value.* = std.math.cast(u8, item) orelse return error.InvalidTensorDataValue;
    }
}

fn fillBoolFromInt32Data(out: []bool, items: []const i32) !void {
    if (items.len != out.len) return error.TensorElementCountMismatch;

    for (out, items) |*value, item| {
        value.* = item != 0;
    }
}

fn intAttr(node: *const onnx.NodeProto, name: []const u8, default: i64) !i64 {
    for (node.attribute.items) |*attribute| {
        if (attribute.name) |attribute_name| {
            if (std.mem.eql(u8, attribute_name, name)) {
                return attribute.i orelse error.InvalidIntegerAttribute;
            }
        }
    }

    return default;
}

fn floatAttr(node: *const onnx.NodeProto, name: []const u8, default: f32) !f32 {
    for (node.attribute.items) |*attribute| {
        if (attribute.name) |attribute_name| {
            if (std.mem.eql(u8, attribute_name, name)) {
                return attribute.f orelse error.InvalidFloatAttribute;
            }
        }
    }

    return default;
}

fn pairAttr(node: *const onnx.NodeProto, name: []const u8, default: [2]usize) ![2]usize {
    for (node.attribute.items) |*attribute| {
        if (attribute.name) |attribute_name| {
            if (std.mem.eql(u8, attribute_name, name)) {
                if (attribute.ints.items.len != 2) return error.InvalidAttributeLength;
                return .{
                    std.math.cast(usize, attribute.ints.items[0]) orelse return error.InvalidAttributeValue,
                    std.math.cast(usize, attribute.ints.items[1]) orelse return error.InvalidAttributeValue,
                };
            }
        }
    }

    return default;
}

fn padsAttr(node: *const onnx.NodeProto) ![4]usize {
    for (node.attribute.items) |*attribute| {
        if (attribute.name) |attribute_name| {
            if (std.mem.eql(u8, attribute_name, "auto_pad")) {
                if (attribute.s) |value| {
                    if (!std.mem.eql(u8, value, "NOTSET")) return error.ConvAutoPadUnsupported;
                }
            }

            if (std.mem.eql(u8, attribute_name, "pads")) {
                if (attribute.ints.items.len != 4) return error.InvalidAttributeLength;
                return .{
                    std.math.cast(usize, attribute.ints.items[0]) orelse return error.InvalidAttributeValue,
                    std.math.cast(usize, attribute.ints.items[1]) orelse return error.InvalidAttributeValue,
                    std.math.cast(usize, attribute.ints.items[2]) orelse return error.InvalidAttributeValue,
                    std.math.cast(usize, attribute.ints.items[3]) orelse return error.InvalidAttributeValue,
                };
            }
        }
    }

    return .{ 0, 0, 0, 0 };
}

fn ensureDefaultDomain(node: *const onnx.NodeProto) !void {
    if (node.domain) |domain| {
        if (domain.len != 0) return error.UnsupportedOperatorDomain;
    }
}
