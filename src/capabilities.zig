const std = @import("std");
const onnx = @import("proto/onnx.pb.zig");

pub const OpSet = struct {
    domain: []const u8,
    version: ?i64,
};

pub const OperatorCount = struct {
    domain: []const u8,
    op_type: []const u8,
    count: usize,
    supported: bool,
};

pub const IssueKind = enum {
    missing_graph,
    missing_op_type,
    unsupported_domain,
    unsupported_operator,
    unsupported_tensor_dtype,
    dynamic_dimension,
    invalid_dimension,
    missing_tensor_shape,
    external_initializer,
    sparse_initializer,
    multi_output_node,
};

pub const Issue = struct {
    kind: IssueKind,
    subject: []const u8,
    detail: []const u8 = "",
};

pub const Report = struct {
    opsets: std.ArrayList(OpSet) = .empty,
    operators: std.ArrayList(OperatorCount) = .empty,
    issues: std.ArrayList(Issue) = .empty,

    pub fn deinit(self: *Report, allocator: std.mem.Allocator) void {
        self.opsets.deinit(allocator);
        self.operators.deinit(allocator);
        self.issues.deinit(allocator);
        self.* = undefined;
    }

    pub fn supported(self: Report) bool {
        return self.issues.items.len == 0;
    }
};

pub fn analyze(allocator: std.mem.Allocator, model: *const onnx.ModelProto) !Report {
    var report: Report = .{};
    errdefer report.deinit(allocator);

    for (model.opset_import.items) |*opset| {
        try report.opsets.append(allocator, .{
            .domain = normalizeDomain(opset.domain),
            .version = opset.version,
        });
    }

    const graph = if (model.graph) |*graph| graph else {
        try report.issues.append(allocator, .{
            .kind = .missing_graph,
            .subject = "model",
        });
        return report;
    };

    if (graph.sparse_initializer.items.len != 0) {
        try report.issues.append(allocator, .{
            .kind = .sparse_initializer,
            .subject = graph.name orelse "graph",
            .detail = "sparse initializers are not supported",
        });
    }

    for (graph.initializer.items) |*initializer| {
        try inspectInitializer(allocator, &report, initializer);
    }

    for (graph.input.items) |*input| {
        try inspectValueInfo(allocator, &report, input);
    }

    for (graph.output.items) |*output| {
        try inspectValueInfo(allocator, &report, output);
    }

    for (graph.value_info.items) |*value_info| {
        try inspectValueInfo(allocator, &report, value_info);
    }

    for (graph.node.items) |*node| {
        try inspectNode(allocator, &report, node);
    }

    return report;
}

pub fn print(report: Report) void {
    std.debug.print("ONNX capability report\n", .{});

    std.debug.print("opsets: {d}\n", .{report.opsets.items.len});
    for (report.opsets.items) |opset| {
        std.debug.print("  {s}: ", .{opset.domain});
        if (opset.version) |version| {
            std.debug.print("{d}\n", .{version});
        } else {
            std.debug.print("<missing>\n", .{});
        }
    }

    std.debug.print("operators: {d}\n", .{report.operators.items.len});
    for (report.operators.items) |operator| {
        const marker = if (operator.supported) "supported" else "unsupported";
        std.debug.print("  {s}::{s} x{d} ({s})\n", .{
            operator.domain,
            operator.op_type,
            operator.count,
            marker,
        });
    }

    std.debug.print("issues: {d}\n", .{report.issues.items.len});
    for (report.issues.items) |issue| {
        std.debug.print("  {s}: {s}", .{ @tagName(issue.kind), issue.subject });
        if (issue.detail.len != 0) {
            std.debug.print(" - {s}", .{issue.detail});
        }
        std.debug.print("\n", .{});
    }
}

fn inspectNode(allocator: std.mem.Allocator, report: *Report, node: *const onnx.NodeProto) !void {
    const op_type = node.op_type orelse {
        try report.issues.append(allocator, .{
            .kind = .missing_op_type,
            .subject = node.name orelse "<unnamed node>",
        });
        return;
    };

    const domain = normalizeDomain(node.domain);
    const supported = isSupportedOperator(domain, op_type);

    try addOperator(report, allocator, domain, op_type, supported);

    if (!isDefaultDomain(domain)) {
        try report.issues.append(allocator, .{
            .kind = .unsupported_domain,
            .subject = op_type,
            .detail = domain,
        });
    } else if (!supported) {
        try report.issues.append(allocator, .{
            .kind = .unsupported_operator,
            .subject = op_type,
        });
    }

    if (node.output.items.len != 1) {
        try report.issues.append(allocator, .{
            .kind = .multi_output_node,
            .subject = node.name orelse op_type,
            .detail = "executor currently expects one output per node",
        });
    }
}

fn inspectInitializer(allocator: std.mem.Allocator, report: *Report, initializer: *const onnx.TensorProto) !void {
    const name = initializer.name orelse "<unnamed initializer>";

    try inspectTensorDataType(allocator, report, name, initializer.data_type);

    if (initializer.data_location) |location| {
        if (location == .EXTERNAL) {
            try report.issues.append(allocator, .{
                .kind = .external_initializer,
                .subject = name,
                .detail = "external tensor data is not supported",
            });
        }
    }

    for (initializer.dims.items) |dim| {
        if (dim <= 0) {
            try report.issues.append(allocator, .{
                .kind = .invalid_dimension,
                .subject = name,
            });
            return;
        }
    }
}

fn inspectValueInfo(allocator: std.mem.Allocator, report: *Report, value_info: *const onnx.ValueInfoProto) !void {
    const name = value_info.name orelse "<unnamed value>";
    const type_proto = value_info.type orelse return;
    const type_value = type_proto.value orelse return;

    switch (type_value) {
        .tensor_type => |tensor_type| {
            try inspectTensorDataType(allocator, report, name, tensor_type.elem_type);

            const shape = tensor_type.shape orelse {
                try report.issues.append(allocator, .{
                    .kind = .missing_tensor_shape,
                    .subject = name,
                });
                return;
            };

            for (shape.dim.items) |dim| {
                const value = dim.value orelse {
                    try report.issues.append(allocator, .{
                        .kind = .dynamic_dimension,
                        .subject = name,
                        .detail = "dimension has no value",
                    });
                    continue;
                };

                switch (value) {
                    .dim_value => |dim_value| {
                        if (dim_value <= 0) {
                            try report.issues.append(allocator, .{
                                .kind = .invalid_dimension,
                                .subject = name,
                            });
                        }
                    },
                    .dim_param => |dim_param| {
                        try report.issues.append(allocator, .{
                            .kind = .dynamic_dimension,
                            .subject = name,
                            .detail = dim_param,
                        });
                    },
                }
            }
        },
        else => {
            try report.issues.append(allocator, .{
                .kind = .unsupported_tensor_dtype,
                .subject = name,
                .detail = "non-tensor value type",
            });
        },
    }
}

fn inspectTensorDataType(
    allocator: std.mem.Allocator,
    report: *Report,
    subject: []const u8,
    data_type: ?i32,
) !void {
    if (isSupportedTensorDataType(data_type)) return;

    try report.issues.append(allocator, .{
        .kind = .unsupported_tensor_dtype,
        .subject = subject,
        .detail = tensorDataTypeName(data_type),
    });
}

fn addOperator(
    report: *Report,
    allocator: std.mem.Allocator,
    domain: []const u8,
    op_type: []const u8,
    supported: bool,
) !void {
    for (report.operators.items) |*operator| {
        if (std.mem.eql(u8, operator.domain, domain) and
            std.mem.eql(u8, operator.op_type, op_type))
        {
            operator.count += 1;
            operator.supported = operator.supported and supported;
            return;
        }
    }

    try report.operators.append(allocator, .{
        .domain = domain,
        .op_type = op_type,
        .count = 1,
        .supported = supported,
    });
}

fn normalizeDomain(domain: ?[]const u8) []const u8 {
    const value = domain orelse "";
    if (value.len == 0) return "ai.onnx";
    return value;
}

fn isDefaultDomain(domain: []const u8) bool {
    return std.mem.eql(u8, domain, "ai.onnx");
}

fn isSupportedOperator(domain: []const u8, op_type: []const u8) bool {
    if (!isDefaultDomain(domain)) return false;

    return std.mem.eql(u8, op_type, "Add") or
        std.mem.eql(u8, op_type, "Constant") or
        std.mem.eql(u8, op_type, "Conv") or
        std.mem.eql(u8, op_type, "Flatten") or
        std.mem.eql(u8, op_type, "Gemm") or
        std.mem.eql(u8, op_type, "MatMul") or
        std.mem.eql(u8, op_type, "Softmax");
}

fn isSupportedTensorDataType(data_type: ?i32) bool {
    const actual = data_type orelse return false;
    return actual == @intFromEnum(onnx.TensorProto.DataType.FLOAT) or
        actual == @intFromEnum(onnx.TensorProto.DataType.INT64) or
        actual == @intFromEnum(onnx.TensorProto.DataType.INT32) or
        actual == @intFromEnum(onnx.TensorProto.DataType.UINT8) or
        actual == @intFromEnum(onnx.TensorProto.DataType.BOOL);
}

fn tensorDataTypeName(data_type: ?i32) []const u8 {
    const actual = data_type orelse return "<missing>";

    if (actual == @intFromEnum(onnx.TensorProto.DataType.UNDEFINED)) return "UNDEFINED";
    if (actual == @intFromEnum(onnx.TensorProto.DataType.FLOAT)) return "FLOAT";
    if (actual == @intFromEnum(onnx.TensorProto.DataType.UINT8)) return "UINT8";
    if (actual == @intFromEnum(onnx.TensorProto.DataType.INT8)) return "INT8";
    if (actual == @intFromEnum(onnx.TensorProto.DataType.UINT16)) return "UINT16";
    if (actual == @intFromEnum(onnx.TensorProto.DataType.INT16)) return "INT16";
    if (actual == @intFromEnum(onnx.TensorProto.DataType.INT32)) return "INT32";
    if (actual == @intFromEnum(onnx.TensorProto.DataType.INT64)) return "INT64";
    if (actual == @intFromEnum(onnx.TensorProto.DataType.STRING)) return "STRING";
    if (actual == @intFromEnum(onnx.TensorProto.DataType.BOOL)) return "BOOL";
    if (actual == @intFromEnum(onnx.TensorProto.DataType.FLOAT16)) return "FLOAT16";
    if (actual == @intFromEnum(onnx.TensorProto.DataType.DOUBLE)) return "DOUBLE";
    if (actual == @intFromEnum(onnx.TensorProto.DataType.UINT32)) return "UINT32";
    if (actual == @intFromEnum(onnx.TensorProto.DataType.UINT64)) return "UINT64";
    if (actual == @intFromEnum(onnx.TensorProto.DataType.COMPLEX64)) return "COMPLEX64";
    if (actual == @intFromEnum(onnx.TensorProto.DataType.COMPLEX128)) return "COMPLEX128";
    if (actual == @intFromEnum(onnx.TensorProto.DataType.BFLOAT16)) return "BFLOAT16";

    return "<unknown>";
}
