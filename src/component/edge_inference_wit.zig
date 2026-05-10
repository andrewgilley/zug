const wit = @import("wit.zig");

const bool_type = wit.Type.primitiveType(.bool);
const u8_type = wit.Type.primitiveType(.u8);
const u32_type = wit.Type.primitiveType(.u32);
const u64_type = wit.Type.primitiveType(.u64);
const string_type = wit.Type.primitiveType(.string);

const byte_list_type = wit.Type{ .list = &u8_type };
const u64_list_type = wit.Type{ .list = &u64_type };
const string_list_type = wit.Type{ .list = &string_type };

const dtype_type = wit.Type.namedType("dtype");
const tensor_desc_type = wit.Type.namedType("tensor-desc");
const tensor_type = wit.Type.namedType("tensor");
const graph_encoding_type = wit.Type.namedType("graph-encoding");
const execution_target_type = wit.Type.namedType("execution-target");
const inference_error_type = wit.Type.namedType("inference-error");
const world_inference_error_type = wit.Type.namedType("inference.inference-error");
const graph_type = wit.Type.namedType("graph");
const context_type = wit.Type.namedType("context");
const target_profile_type = wit.Type.namedType("target-profile");
const compatibility_report_type = wit.Type.namedType("compatibility-report");

const graph_result_type = wit.Type{ .result = .{ .ok = &graph_type, .err = &inference_error_type } };
const context_result_type = wit.Type{ .result = .{ .ok = &context_type, .err = &inference_error_type } };
const unit_result_type = wit.Type{ .result = .{ .err = &inference_error_type } };
const world_unit_result_type = wit.Type{ .result = .{ .err = &world_inference_error_type } };
const tensor_desc_result_type = wit.Type{ .result = .{ .ok = &tensor_desc_type, .err = &inference_error_type } };
const tensor_result_type = wit.Type{ .result = .{ .ok = &tensor_type, .err = &inference_error_type } };

pub const package = wit.Package{
    .id = "zug:edge-inference@0.1.0",
    .interfaces = &.{
        tensors_interface,
        inference_interface,
        diagnostics_interface,
        log_interface,
    },
    .worlds = &.{edge_guest_world},
};

pub const tensors_interface = wit.Interface{
    .name = "tensors",
    .declarations = &.{
        .{ .@"enum" = .{
            .name = "dtype",
            .cases = &.{ "float32", "int64", "int32", "uint8", "bool" },
        } },
        .{ .record = .{
            .name = "tensor-desc",
            .fields = &.{
                .{ .name = "dtype", .ty = dtype_type },
                .{ .name = "shape", .ty = u64_list_type },
                .{ .name = "byte-len", .ty = u64_type },
            },
        } },
        .{ .record = .{
            .name = "tensor",
            .fields = &.{
                .{ .name = "desc", .ty = tensor_desc_type },
                .{ .name = "data", .ty = byte_list_type },
            },
        } },
    },
};

pub const inference_interface = wit.Interface{
    .name = "inference",
    .uses = &.{.{
        .interface_name = "tensors",
        .names = &.{ "tensor", "tensor-desc" },
    }},
    .declarations = &.{
        .{ .@"enum" = .{
            .name = "graph-encoding",
            .cases = &.{"onnx"},
        } },
        .{ .@"enum" = .{
            .name = "execution-target",
            .cases = &.{"cpu"},
        } },
        .{ .variant = .{
            .name = "inference-error",
            .cases = &.{
                .{ .name = "invalid-encoding" },
                .{ .name = "invalid-target" },
                .{ .name = "invalid-graph-handle" },
                .{ .name = "invalid-context-handle" },
                .{ .name = "invalid-input" },
                .{ .name = "unsupported-dtype" },
                .{ .name = "unsupported-op", .payload = string_type },
                .{ .name = "buffer-too-small" },
                .{ .name = "model-too-large" },
                .{ .name = "missing-model" },
                .{ .name = "runtime-error", .payload = string_type },
            },
        } },
        .{ .alias = .{ .name = "graph", .target = u32_type } },
        .{ .alias = .{ .name = "context", .target = u32_type } },
    },
    .functions = &.{
        .{
            .name = "load-graph",
            .params = &.{
                .{ .name = "model", .ty = byte_list_type },
                .{ .name = "encoding", .ty = graph_encoding_type },
                .{ .name = "target", .ty = execution_target_type },
            },
            .result = graph_result_type,
        },
        .{
            .name = "load-preloaded-graph",
            .params = &.{
                .{ .name = "encoding", .ty = graph_encoding_type },
                .{ .name = "target", .ty = execution_target_type },
            },
            .result = graph_result_type,
        },
        .{
            .name = "init-execution-context",
            .params = &.{.{ .name = "graph", .ty = graph_type }},
            .result = context_result_type,
        },
        .{
            .name = "set-input",
            .params = &.{
                .{ .name = "context", .ty = context_type },
                .{ .name = "index", .ty = u32_type },
                .{ .name = "value", .ty = tensor_type },
            },
            .result = unit_result_type,
        },
        .{
            .name = "compute",
            .params = &.{.{ .name = "context", .ty = context_type }},
            .result = unit_result_type,
        },
        .{
            .name = "output-desc",
            .params = &.{
                .{ .name = "context", .ty = context_type },
                .{ .name = "index", .ty = u32_type },
            },
            .result = tensor_desc_result_type,
        },
        .{
            .name = "output",
            .params = &.{
                .{ .name = "context", .ty = context_type },
                .{ .name = "index", .ty = u32_type },
            },
            .result = tensor_result_type,
        },
    },
};

pub const diagnostics_interface = wit.Interface{
    .name = "diagnostics",
    .uses = &.{.{
        .interface_name = "inference",
        .names = &.{ "graph-encoding", "execution-target", "inference-error" },
    }},
    .declarations = &.{
        .{ .record = .{
            .name = "target-profile",
            .fields = &.{
                .{ .name = "name", .ty = string_type },
                .{ .name = "max-model-bytes", .ty = u64_type },
                .{ .name = "max-memory-pages", .ty = u32_type },
                .{ .name = "supported-ops", .ty = string_list_type },
                .{ .name = "supported-dtypes", .ty = string_list_type },
            },
        } },
        .{ .record = .{
            .name = "compatibility-report",
            .fields = &.{
                .{ .name = "compatible", .ty = bool_type },
                .{ .name = "profile", .ty = target_profile_type },
                .{ .name = "errors", .ty = wit.Type{ .list = &inference_error_type } },
                .{ .name = "warnings", .ty = string_list_type },
            },
        } },
    },
    .functions = &.{.{
        .name = "inspect-model",
        .params = &.{
            .{ .name = "model", .ty = byte_list_type },
            .{ .name = "encoding", .ty = graph_encoding_type },
            .{ .name = "target", .ty = execution_target_type },
            .{ .name = "profile", .ty = target_profile_type },
        },
        .result = compatibility_report_type,
    }},
};

pub const log_interface = wit.Interface{
    .name = "log",
    .functions = &.{.{
        .name = "write",
        .params = &.{.{ .name = "message", .ty = string_type }},
    }},
};

pub const edge_guest_world = wit.World{
    .name = "edge-guest",
    .imports = &.{
        .{ .interface = "inference" },
        .{ .interface = "diagnostics" },
        .{ .interface = "log" },
    },
    .exports = &.{.{
        .function = .{
            .name = "run",
            .result = world_unit_result_type,
        },
    }},
};

test "edge inference descriptor renders WIT" {
    const allocator = @import("std").testing.allocator;
    const rendered = try wit.renderAlloc(allocator, package);
    defer allocator.free(rendered);

    try @import("std").testing.expect(@import("std").mem.startsWith(u8, rendered, "package zug:edge-inference@0.1.0;\n"));
    try @import("std").testing.expect(@import("std").mem.indexOf(u8, rendered, "world edge-guest") != null);
    try @import("std").testing.expect(@import("std").mem.indexOf(u8, rendered, "load-preloaded-graph: func(") != null);
}
