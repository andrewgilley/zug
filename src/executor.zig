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

        const result = if (std.mem.eql(u8, op_type, "Flatten")) blk: {
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
    try requireFloat32(initializer.data_type);

    if (initializer.data_location) |location| {
        if (location != .DEFAULT) return error.ExternalTensorDataUnsupported;
    }

    if (initializer.segment != null) return error.TensorSegmentsUnsupported;

    const shape = try shapeFromDims(allocator, initializer.dims.items);
    errdefer allocator.free(shape);

    const count = try tensor.elementCount(shape);
    const data = try allocator.alloc(f32, count);
    errdefer allocator.free(data);

    if (initializer.raw_data) |raw_data| {
        try fillFloat32FromRawData(data, raw_data);
    } else {
        if (initializer.float_data.items.len != count) return error.TensorElementCountMismatch;
        @memcpy(data, initializer.float_data.items);
    }

    return .{
        .dtype = .float32,
        .shape = shape,
        .data = data,
    };
}

fn tensorFromRawFile(
    allocator: std.mem.Allocator,
    input: *const onnx.ValueInfoProto,
    path: []const u8,
) !tensor.Tensor {
    const shape = try shapeFromValueInfo(allocator, input);
    errdefer allocator.free(shape);

    const count = try tensor.elementCount(shape);
    const expected_bytes = try std.math.mul(usize, count, @sizeOf(f32));

    const bytes = try std.Io.Dir.cwd().readFileAlloc(
        std.Options.debug_io,
        path,
        allocator,
        .limited(max_tensor_file_bytes),
    );
    defer allocator.free(bytes);

    if (bytes.len != expected_bytes) return error.InputByteLengthMismatch;

    const data = try allocator.alloc(f32, count);
    errdefer allocator.free(data);

    try fillFloat32FromRawData(data, bytes);

    return .{
        .dtype = .float32,
        .shape = shape,
        .data = data,
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

fn shapeFromValueInfo(allocator: std.mem.Allocator, input: *const onnx.ValueInfoProto) ![]usize {
    const type_proto = input.type orelse return error.MissingInputType;
    const type_value = type_proto.value orelse return error.MissingInputType;

    const tensor_type = switch (type_value) {
        .tensor_type => |value| value,
        else => return error.UnsupportedInputType,
    };

    try requireFloat32(tensor_type.elem_type);

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

fn ensureDefaultDomain(node: *const onnx.NodeProto) !void {
    if (node.domain) |domain| {
        if (domain.len != 0) return error.UnsupportedOperatorDomain;
    }
}
