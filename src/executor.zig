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

        if (std.mem.eql(u8, op_type, "Split")) {
            try self.runSplit(node);
            return;
        }

        if (std.mem.eql(u8, op_type, "TopK")) {
            try self.runTopK(node);
            return;
        }

        if (node.output.items.len < 1 or node.output.items[0].len == 0) {
            return error.InvalidNodeOutput;
        }
        for (node.output.items[1..]) |extra_output| {
            if (extra_output.len != 0) return error.MultipleNodeOutputsUnsupported;
        }

        const output_name = node.output.items[0];

        const result = if (std.mem.eql(u8, op_type, "Add")) blk: {
            const a = try self.requiredNodeInput(node, 0);
            const b = try self.requiredNodeInput(node, 1);
            break :blk try ops.add(self.allocator, a, b);
        } else if (std.mem.eql(u8, op_type, "AveragePool")) blk: {
            const input = try self.requiredNodeInput(node, 0);
            break :blk try ops.averagePool(self.allocator, input, .{
                .kernel_shape = try requiredPairAttr(node, "kernel_shape"),
                .pads = try padsAttr(node),
                .strides = try pairAttr(node, "strides", .{ 1, 1 }),
                .ceil_mode = (try intAttr(node, "ceil_mode", 0)) != 0,
                .count_include_pad = (try intAttr(node, "count_include_pad", 0)) != 0,
            });
        } else if (std.mem.eql(u8, op_type, "BatchNormalization")) blk: {
            const input = try self.requiredNodeInput(node, 0);
            const scale = try self.requiredNodeInput(node, 1);
            const bias = try self.requiredNodeInput(node, 2);
            const mean = try self.requiredNodeInput(node, 3);
            const variance = try self.requiredNodeInput(node, 4);
            break :blk try ops.batchNormalization(self.allocator, input, scale, bias, mean, variance, .{
                .epsilon = try floatAttr(node, "epsilon", 0.00001),
            });
        } else if (std.mem.eql(u8, op_type, "Cast")) blk: {
            const input = try self.requiredNodeInput(node, 0);
            const to = std.math.cast(i32, try requiredIntAttr(node, "to")) orelse return error.InvalidTensorDataType;
            break :blk try ops.cast(self.allocator, input, try dtypeFromOnnx(to));
        } else if (std.mem.eql(u8, op_type, "Constant")) blk: {
            break :blk try ops.constant(self.allocator, node);
        } else if (std.mem.eql(u8, op_type, "ConstantOfShape")) blk: {
            const shape = try self.requiredNodeInput(node, 0);
            break :blk try ops.constantOfShape(self.allocator, shape, try optionalTensorAttr(node, "value"));
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
        } else if (std.mem.eql(u8, op_type, "Clip")) blk: {
            const input = try self.requiredNodeInput(node, 0);
            const min_input = try self.optionalNodeInput(node, 1);
            const max_input = try self.optionalNodeInput(node, 2);
            break :blk try ops.clip(
                self.allocator,
                input,
                min_input,
                max_input,
                try optionalFloatAttr(node, "min"),
                try optionalFloatAttr(node, "max"),
            );
        } else if (std.mem.eql(u8, op_type, "Concat")) blk: {
            const inputs = try self.nodeInputs(node);
            defer self.allocator.free(inputs);
            const axis = try intAttr(node, "axis", 0);
            break :blk try ops.concat(self.allocator, inputs, axis);
        } else if (std.mem.eql(u8, op_type, "Div")) blk: {
            const a = try self.requiredNodeInput(node, 0);
            const b = try self.requiredNodeInput(node, 1);
            break :blk try ops.div(self.allocator, a, b);
        } else if (std.mem.eql(u8, op_type, "Equal")) blk: {
            const a = try self.requiredNodeInput(node, 0);
            const b = try self.requiredNodeInput(node, 1);
            break :blk try ops.equal(self.allocator, a, b);
        } else if (std.mem.eql(u8, op_type, "Expand")) blk: {
            const input = try self.requiredNodeInput(node, 0);
            const shape = try self.requiredNodeInput(node, 1);
            break :blk try ops.expand(self.allocator, input, shape);
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
        } else if (std.mem.eql(u8, op_type, "Gather")) blk: {
            const data = try self.requiredNodeInput(node, 0);
            const indices = try self.requiredNodeInput(node, 1);
            const axis = try intAttr(node, "axis", 0);
            break :blk try ops.gather(self.allocator, data, indices, axis);
        } else if (std.mem.eql(u8, op_type, "GlobalAveragePool")) blk: {
            const input = try self.requiredNodeInput(node, 0);
            break :blk try ops.globalAveragePool(self.allocator, input);
        } else if (std.mem.eql(u8, op_type, "Greater")) blk: {
            const a = try self.requiredNodeInput(node, 0);
            const b = try self.requiredNodeInput(node, 1);
            break :blk try ops.greater(self.allocator, a, b);
        } else if (std.mem.eql(u8, op_type, "Identity")) blk: {
            const input = try self.requiredNodeInput(node, 0);
            break :blk try ops.identity(self.allocator, input);
        } else if (std.mem.eql(u8, op_type, "LeakyRelu")) blk: {
            const input = try self.requiredNodeInput(node, 0);
            break :blk try ops.leakyRelu(self.allocator, input, try floatAttr(node, "alpha", 0.01));
        } else if (std.mem.eql(u8, op_type, "Less")) blk: {
            const a = try self.requiredNodeInput(node, 0);
            const b = try self.requiredNodeInput(node, 1);
            break :blk try ops.less(self.allocator, a, b);
        } else if (std.mem.eql(u8, op_type, "MatMul")) blk: {
            const a = try self.requiredNodeInput(node, 0);
            const b = try self.requiredNodeInput(node, 1);
            break :blk try ops.matmul(self.allocator, a, b);
        } else if (std.mem.eql(u8, op_type, "MaxPool")) blk: {
            const input = try self.requiredNodeInput(node, 0);
            break :blk try ops.maxPool(self.allocator, input, .{
                .kernel_shape = try requiredPairAttr(node, "kernel_shape"),
                .pads = try padsAttr(node),
                .strides = try pairAttr(node, "strides", .{ 1, 1 }),
                .ceil_mode = (try intAttr(node, "ceil_mode", 0)) != 0,
            });
        } else if (std.mem.eql(u8, op_type, "Mul")) blk: {
            const a = try self.requiredNodeInput(node, 0);
            const b = try self.requiredNodeInput(node, 1);
            break :blk try ops.mul(self.allocator, a, b);
        } else if (std.mem.eql(u8, op_type, "Pad")) blk: {
            const input = try self.requiredNodeInput(node, 0);
            const pads = try self.requiredNodeInput(node, 1);
            const constant_value = try self.optionalNodeInput(node, 2);
            if (try self.optionalNodeInput(node, 3) != null) return error.PadAxesUnsupported;
            break :blk try ops.pad(self.allocator, input, pads, constant_value, .{
                .mode = try padModeAttr(node),
            });
        } else if (std.mem.eql(u8, op_type, "Pow")) blk: {
            const a = try self.requiredNodeInput(node, 0);
            const b = try self.requiredNodeInput(node, 1);
            break :blk try ops.pow(self.allocator, a, b);
        } else if (std.mem.eql(u8, op_type, "Reciprocal")) blk: {
            const input = try self.requiredNodeInput(node, 0);
            break :blk try ops.reciprocal(self.allocator, input);
        } else if (std.mem.eql(u8, op_type, "Relu")) blk: {
            const input = try self.requiredNodeInput(node, 0);
            break :blk try ops.relu(self.allocator, input);
        } else if (std.mem.eql(u8, op_type, "ReduceMax")) blk: {
            const input = try self.requiredNodeInput(node, 0);
            var axes: ?[]i64 = null;
            if (try self.optionalNodeInput(node, 1)) |axes_tensor| {
                axes = try tensorIntList(self.allocator, axes_tensor);
            } else {
                axes = try optionalIntListAttr(self.allocator, node, "axes");
            }
            defer if (axes) |items| self.allocator.free(items);

            break :blk try ops.reduceMax(
                self.allocator,
                input,
                axes,
                (try intAttr(node, "keepdims", 1)) != 0,
                (try intAttr(node, "noop_with_empty_axes", 0)) != 0,
            );
        } else if (std.mem.eql(u8, op_type, "ReduceMean")) blk: {
            const input = try self.requiredNodeInput(node, 0);
            var axes: ?[]i64 = null;
            if (try self.optionalNodeInput(node, 1)) |axes_tensor| {
                axes = try tensorIntList(self.allocator, axes_tensor);
            } else {
                axes = try optionalIntListAttr(self.allocator, node, "axes");
            }
            defer if (axes) |items| self.allocator.free(items);

            break :blk try ops.reduceMean(
                self.allocator,
                input,
                axes,
                (try intAttr(node, "keepdims", 1)) != 0,
                (try intAttr(node, "noop_with_empty_axes", 0)) != 0,
            );
        } else if (std.mem.eql(u8, op_type, "ReduceSum")) blk: {
            const input = try self.requiredNodeInput(node, 0);
            var axes: ?[]i64 = null;
            if (try self.optionalNodeInput(node, 1)) |axes_tensor| {
                axes = try tensorIntList(self.allocator, axes_tensor);
            } else {
                axes = try optionalIntListAttr(self.allocator, node, "axes");
            }
            defer if (axes) |items| self.allocator.free(items);

            break :blk try ops.reduceSum(
                self.allocator,
                input,
                axes,
                (try intAttr(node, "keepdims", 1)) != 0,
                (try intAttr(node, "noop_with_empty_axes", 0)) != 0,
            );
        } else if (std.mem.eql(u8, op_type, "Reshape")) blk: {
            const input = try self.requiredNodeInput(node, 0);
            const shape = try self.requiredNodeInput(node, 1);
            const allow_zero = (try intAttr(node, "allowzero", 0)) != 0;
            break :blk try ops.reshape(self.allocator, input, shape, allow_zero);
        } else if (std.mem.eql(u8, op_type, "Resize")) blk: {
            const input = try self.requiredNodeInput(node, 0);
            _ = try self.optionalNodeInput(node, 1);
            const scales = try self.optionalNodeInput(node, 2);
            const sizes = try self.optionalNodeInput(node, 3);
            break :blk try ops.resize(self.allocator, input, scales, sizes, .{
                .mode = try resizeModeAttr(node),
            });
        } else if (std.mem.eql(u8, op_type, "Shape")) blk: {
            const input = try self.requiredNodeInput(node, 0);
            break :blk try ops.shapeTensor(self.allocator, input);
        } else if (std.mem.eql(u8, op_type, "Sigmoid")) blk: {
            const input = try self.requiredNodeInput(node, 0);
            break :blk try ops.sigmoid(self.allocator, input);
        } else if (std.mem.eql(u8, op_type, "Slice")) blk: {
            const data = try self.requiredNodeInput(node, 0);
            const starts = try self.requiredNodeInput(node, 1);
            const ends = try self.requiredNodeInput(node, 2);
            const axes = try self.optionalNodeInput(node, 3);
            const steps = try self.optionalNodeInput(node, 4);
            break :blk try ops.slice(self.allocator, data, starts, ends, axes, steps);
        } else if (std.mem.eql(u8, op_type, "Softmax")) blk: {
            const input = try self.requiredNodeInput(node, 0);
            const axis = try intAttr(node, "axis", -1);
            break :blk try ops.softmax(self.allocator, input, axis);
        } else if (std.mem.eql(u8, op_type, "Sqrt")) blk: {
            const input = try self.requiredNodeInput(node, 0);
            break :blk try ops.sqrt(self.allocator, input);
        } else if (std.mem.eql(u8, op_type, "Sub")) blk: {
            const a = try self.requiredNodeInput(node, 0);
            const b = try self.requiredNodeInput(node, 1);
            break :blk try ops.sub(self.allocator, a, b);
        } else if (std.mem.eql(u8, op_type, "Squeeze")) blk: {
            const input = try self.requiredNodeInput(node, 0);
            var axes: ?[]i64 = null;
            if (try self.optionalNodeInput(node, 1)) |axes_tensor| {
                axes = try tensorIntList(self.allocator, axes_tensor);
            } else {
                axes = try optionalIntListAttr(self.allocator, node, "axes");
            }
            defer if (axes) |items| self.allocator.free(items);
            break :blk try ops.squeeze(self.allocator, input, axes);
        } else if (std.mem.eql(u8, op_type, "Tanh")) blk: {
            const input = try self.requiredNodeInput(node, 0);
            break :blk try ops.tanh(self.allocator, input);
        } else if (std.mem.eql(u8, op_type, "Transpose")) blk: {
            const input = try self.requiredNodeInput(node, 0);
            const perm = try optionalUsizeListAttr(self.allocator, node, "perm");
            defer if (perm) |items| self.allocator.free(items);
            break :blk try ops.transpose(self.allocator, input, perm);
        } else if (std.mem.eql(u8, op_type, "Unsqueeze")) blk: {
            const input = try self.requiredNodeInput(node, 0);
            var axes: ?[]i64 = null;
            if (try self.optionalNodeInput(node, 1)) |axes_tensor| {
                axes = try tensorIntList(self.allocator, axes_tensor);
            } else {
                axes = try intListAttr(self.allocator, node, "axes");
            }
            defer if (axes) |items| self.allocator.free(items);
            break :blk try ops.unsqueeze(self.allocator, input, axes.?);
        } else if (std.mem.eql(u8, op_type, "Where")) blk: {
            const condition = try self.requiredNodeInput(node, 0);
            const x = try self.requiredNodeInput(node, 1);
            const y = try self.requiredNodeInput(node, 2);
            break :blk try ops.where(self.allocator, condition, x, y);
        } else {
            return error.UnsupportedOperator;
        };

        try self.put(output_name, result);
    }

    fn runSplit(self: *Executor, node: *const onnx.NodeProto) !void {
        if (node.output.items.len == 0) return error.InvalidNodeOutput;
        for (node.output.items) |output_name| {
            if (output_name.len == 0) return error.InvalidNodeOutput;
        }

        const input = try self.requiredNodeInput(node, 0);
        const split_sizes = try self.optionalNodeInput(node, 1);
        const axis = try intAttr(node, "axis", 0);

        var result = try ops.split(self.allocator, input, split_sizes, axis, node.output.items.len);
        var next_unowned: usize = 0;
        errdefer {
            for (result.outputs[next_unowned..]) |*output| {
                output.deinit(self.allocator);
            }
            self.allocator.free(result.outputs);
        }

        for (node.output.items, 0..) |output_name, index| {
            const value = result.outputs[index];
            result.outputs[index] = undefined;
            next_unowned = index + 1;
            try self.put(output_name, value);
        }

        self.allocator.free(result.outputs);
    }

    fn runTopK(self: *Executor, node: *const onnx.NodeProto) !void {
        if (node.output.items.len != 2 or node.output.items[0].len == 0 or node.output.items[1].len == 0) {
            return error.InvalidNodeOutput;
        }

        const input = try self.requiredNodeInput(node, 0);
        const k = try self.requiredNodeInput(node, 1);

        var result = try ops.topK(
            self.allocator,
            input,
            k,
            try intAttr(node, "axis", -1),
            (try intAttr(node, "largest", 1)) != 0,
            (try intAttr(node, "sorted", 1)) != 0,
        );

        var values_owned = true;
        var indices_owned = true;
        errdefer {
            if (values_owned) result.values.deinit(self.allocator);
            if (indices_owned) result.indices.deinit(self.allocator);
        }

        const values = result.values;
        result.values = undefined;
        values_owned = false;
        try self.put(node.output.items[0], values);

        const indices = result.indices;
        result.indices = undefined;
        indices_owned = false;
        try self.put(node.output.items[1], indices);
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

    fn nodeInputs(self: *Executor, node: *const onnx.NodeProto) ![]*const tensor.Tensor {
        const inputs = try self.allocator.alloc(*const tensor.Tensor, node.input.items.len);
        errdefer self.allocator.free(inputs);

        for (inputs, node.input.items) |*out, input_name| {
            if (input_name.len == 0) return error.MissingNodeInput;
            out.* = self.values.getPtr(input_name) orelse return error.MissingNodeInputValue;
        }

        return inputs;
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
    const bytes = try std.Io.Dir.cwd().readFileAlloc(
        std.Options.debug_io,
        path,
        allocator,
        .limited(max_tensor_file_bytes),
    );
    defer allocator.free(bytes);

    const element_size = elementByteSize(dtype);
    if (bytes.len % element_size != 0) return error.InputByteLengthMismatch;

    const file_element_count = bytes.len / element_size;
    const shape = try shapeFromValueInfo(allocator, input, file_element_count);
    defer allocator.free(shape);

    const expected_count = try tensor.elementCount(shape);
    if (expected_count != file_element_count) return error.InputByteLengthMismatch;

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

fn shapeFromValueInfo(
    allocator: std.mem.Allocator,
    input: *const onnx.ValueInfoProto,
    file_element_count: usize,
) ![]usize {
    const type_proto = input.type orelse return error.MissingInputType;
    const type_value = type_proto.value orelse return error.MissingInputType;

    const tensor_type = switch (type_value) {
        .tensor_type => |value| value,
        else => return error.UnsupportedInputType,
    };

    const shape_proto = tensor_type.shape orelse return error.MissingInputShape;
    const shape = try allocator.alloc(usize, shape_proto.dim.items.len);
    errdefer allocator.free(shape);

    var known_count: usize = 1;
    var dynamic_index: ?usize = null;

    for (shape_proto.dim.items, 0..) |dim, index| {
        const value = dim.value orelse {
            if (dynamic_index != null) return error.MultipleDynamicDimensionsUnsupported;
            dynamic_index = index;
            shape[index] = 1;
            continue;
        };

        switch (value) {
            .dim_value => |dim_value| {
                if (dim_value <= 0) return error.InvalidTensorDimension;
                shape[index] = std.math.cast(usize, dim_value) orelse return error.DimensionTooLarge;
                known_count = try std.math.mul(usize, known_count, shape[index]);
            },
            .dim_param => {
                if (dynamic_index != null) return error.MultipleDynamicDimensionsUnsupported;
                dynamic_index = index;
                shape[index] = 1;
            },
        }
    }

    if (dynamic_index) |index| {
        if (known_count == 0 or file_element_count % known_count != 0) {
            return error.InputByteLengthMismatch;
        }

        const inferred = file_element_count / known_count;
        if (inferred == 0) return error.InvalidTensorDimension;
        shape[index] = inferred;
    } else if (known_count != file_element_count) {
        return error.InputByteLengthMismatch;
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
            const owned_shape = try allocator.dupe(usize, shape);
            errdefer allocator.free(owned_shape);

            const data = try allocator.alloc(f32, count);
            errdefer allocator.free(data);
            try fillFloat32FromRawData(data, raw_data);

            break :blk tensor.Tensor.initOwnedFloat32(allocator, owned_shape, data);
        },
        .int64 => blk: {
            const owned_shape = try allocator.dupe(usize, shape);
            errdefer allocator.free(owned_shape);

            const data = try allocator.alloc(i64, count);
            errdefer allocator.free(data);
            try fillInt64FromRawData(data, raw_data);

            break :blk tensor.Tensor.initOwnedInt64(allocator, owned_shape, data);
        },
        .int32 => blk: {
            const owned_shape = try allocator.dupe(usize, shape);
            errdefer allocator.free(owned_shape);

            const data = try allocator.alloc(i32, count);
            errdefer allocator.free(data);
            try fillInt32FromRawData(data, raw_data);

            break :blk tensor.Tensor.initOwnedInt32(allocator, owned_shape, data);
        },
        .uint8 => tensor.Tensor.initUint8(allocator, shape, raw_data),
        .bool => blk: {
            const owned_shape = try allocator.dupe(usize, shape);
            errdefer allocator.free(owned_shape);

            const data = try allocator.alloc(bool, count);
            errdefer allocator.free(data);
            for (data, raw_data) |*value, raw| {
                value.* = raw != 0;
            }

            break :blk tensor.Tensor.initOwnedBool(allocator, owned_shape, data);
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

fn requiredIntAttr(node: *const onnx.NodeProto, name: []const u8) !i64 {
    for (node.attribute.items) |*attribute| {
        if (attribute.name) |attribute_name| {
            if (std.mem.eql(u8, attribute_name, name)) {
                return attribute.i orelse error.InvalidIntegerAttribute;
            }
        }
    }

    return error.MissingIntegerAttribute;
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

fn optionalFloatAttr(node: *const onnx.NodeProto, name: []const u8) !?f32 {
    for (node.attribute.items) |*attribute| {
        if (attribute.name) |attribute_name| {
            if (std.mem.eql(u8, attribute_name, name)) {
                return attribute.f orelse error.InvalidFloatAttribute;
            }
        }
    }

    return null;
}

fn optionalTensorAttr(node: *const onnx.NodeProto, name: []const u8) !?*const onnx.TensorProto {
    for (node.attribute.items) |*attribute| {
        if (attribute.name) |attribute_name| {
            if (std.mem.eql(u8, attribute_name, name)) {
                return if (attribute.t) |*value| value else error.InvalidTensorAttribute;
            }
        }
    }

    return null;
}

fn stringAttr(node: *const onnx.NodeProto, name: []const u8, default: []const u8) ![]const u8 {
    for (node.attribute.items) |*attribute| {
        if (attribute.name) |attribute_name| {
            if (std.mem.eql(u8, attribute_name, name)) {
                return attribute.s orelse error.InvalidStringAttribute;
            }
        }
    }

    return default;
}

fn padModeAttr(node: *const onnx.NodeProto) !ops.PadMode {
    const mode = try stringAttr(node, "mode", "constant");
    if (std.mem.eql(u8, mode, "constant")) return .constant;

    return error.PadModeUnsupported;
}

fn resizeModeAttr(node: *const onnx.NodeProto) !ops.ResizeMode {
    const mode = try stringAttr(node, "mode", "nearest");
    if (std.mem.eql(u8, mode, "nearest")) return .nearest;

    return error.ResizeModeUnsupported;
}

fn intListAttr(
    allocator: std.mem.Allocator,
    node: *const onnx.NodeProto,
    name: []const u8,
) ![]i64 {
    for (node.attribute.items) |*attribute| {
        if (attribute.name) |attribute_name| {
            if (std.mem.eql(u8, attribute_name, name)) {
                return try allocator.dupe(i64, attribute.ints.items);
            }
        }
    }

    return error.MissingIntegerListAttribute;
}

fn optionalIntListAttr(
    allocator: std.mem.Allocator,
    node: *const onnx.NodeProto,
    name: []const u8,
) !?[]i64 {
    for (node.attribute.items) |*attribute| {
        if (attribute.name) |attribute_name| {
            if (std.mem.eql(u8, attribute_name, name)) {
                return try allocator.dupe(i64, attribute.ints.items);
            }
        }
    }

    return null;
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

fn requiredPairAttr(node: *const onnx.NodeProto, name: []const u8) ![2]usize {
    for (node.attribute.items) |*attribute| {
        if (attribute.name) |attribute_name| {
            if (std.mem.eql(u8, attribute_name, name)) {
                if (attribute.ints.items.len != 2) return error.InvalidAttributeValue;

                return .{
                    std.math.cast(usize, attribute.ints.items[0]) orelse return error.InvalidAttributeValue,
                    std.math.cast(usize, attribute.ints.items[1]) orelse return error.InvalidAttributeValue,
                };
            }
        }
    }

    return error.MissingIntegerListAttribute;
}

fn optionalUsizeListAttr(
    allocator: std.mem.Allocator,
    node: *const onnx.NodeProto,
    name: []const u8,
) !?[]usize {
    for (node.attribute.items) |*attribute| {
        if (attribute.name) |attribute_name| {
            if (std.mem.eql(u8, attribute_name, name)) {
                const items = try allocator.alloc(usize, attribute.ints.items.len);
                errdefer allocator.free(items);

                for (items, attribute.ints.items) |*out, item| {
                    out.* = std.math.cast(usize, item) orelse return error.InvalidAttributeValue;
                }

                return items;
            }
        }
    }

    return null;
}

fn tensorIntList(allocator: std.mem.Allocator, value: *const tensor.Tensor) ![]i64 {
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
