const std = @import("std");
const capabilities = @import("capabilities.zig");
const wasm_imports = @import("wasm/imports.zig");
const wasm_manifest = @import("wasm/manifest.zig");

pub const runtime_name = "zug-edge-runtime";
pub const runtime_scope_version = "0.1";

pub const max_model_bytes: usize = 100 * 1024 * 1024;
pub const max_wasm_bytes: usize = 100 * 1024 * 1024;
pub const default_wasm_memory_bytes: usize = 16 * 1024 * 1024;
pub const max_wasm_memory_bytes: usize = (wasm_manifest.RuntimeCapabilities{}).max_memory_bytes;

pub const wasm_core_scope = [_][]const u8{
    "binary module parsing for core MVP sections",
    "single linear memory with bounds-checked loads/stores",
    "tables and call_indirect for function dispatch",
    "globals, start functions, locals, blocks, loops, if/else",
    "branching, br_table, return, select and typed select",
    "i32/i64/f32/f64 numeric, compare, conversion, reinterpret and extend ops",
    "memory.size, memory.grow, memory.copy and memory.fill",
    "module validation for supported types, limits, imports, data and element segments",
};

pub const wasi_preview1_scope = [_][]const u8{
    "wasi_snapshot_preview1.args_sizes_get",
    "wasi_snapshot_preview1.args_get",
    "wasi_snapshot_preview1.environ_sizes_get",
    "wasi_snapshot_preview1.environ_get",
    "wasi_snapshot_preview1.clock_time_get",
    "wasi_snapshot_preview1.random_get",
    "wasi_snapshot_preview1.fd_fdstat_get",
    "wasi_snapshot_preview1.fd_write",
    "wasi_snapshot_preview1.proc_exit",
};

pub const wasi_nn_scope = [_][]const u8{
    "wasi_nn.load_graph",
    "wasi_nn.init_execution_context",
    "wasi_nn.set_input_by_index",
    "wasi_nn.compute",
    "wasi_nn.get_output_descriptor",
    "wasi_nn.get_output",
};

pub const zug_nn_scope = [_][]const u8{
    "zug_nn.load_preloaded_graph",
};

pub const model_execution_scope = [_][]const u8{
    "ONNX graph decoding through generated protobuf bindings",
    "CPU execution target",
    "f32 inference path for WASI-NN guest inputs",
    "direct host ONNX execution through Session and Executor",
    "raw tensor file input and raw output/expectation checks",
    "runtime benchmark reporting for timing and allocation pressure",
};

pub const next_capability_frontiers = [_][]const u8{
    "scratch-buffer reuse for steady-state model execution",
    "specialized Conv kernels for MobileNet-style depthwise and pointwise convolutions",
    "compiled ONNX execution plan with numeric tensor slots instead of per-node string lookup",
    "broader WASI Preview 1 filesystem/preopen surface",
    "official wasm spec-test ingestion for validation and interpreter coverage",
    "additional modern model gates for YOLO-style vision and small transformer graphs",
};

pub fn print() void {
    std.debug.print("{s} scope {s}\n", .{ runtime_name, runtime_scope_version });
    std.debug.print("resource limits:\n", .{});
    std.debug.print("  max_model_bytes: {d}\n", .{max_model_bytes});
    std.debug.print("  max_wasm_bytes: {d}\n", .{max_wasm_bytes});
    std.debug.print("  default_wasm_memory_bytes: {d}\n", .{default_wasm_memory_bytes});
    std.debug.print("  max_wasm_memory_bytes: {d}\n", .{max_wasm_memory_bytes});

    printList("onnx operators", &capabilities.supported_operator_names);
    printList("onnx tensor dtypes", &capabilities.supported_tensor_dtype_names);
    printList("model execution", &model_execution_scope);
    printList("wasm core", &wasm_core_scope);
    printList("wasi preview1 imports", &wasi_preview1_scope);
    printList("wasi-nn imports", &wasi_nn_scope);
    printList("zug host extension imports", &zug_nn_scope);
    printList("next capability frontiers", &next_capability_frontiers);
}

fn printList(title: []const u8, items: []const []const u8) void {
    std.debug.print("{s}: {d}\n", .{ title, items.len });
    for (items) |item| {
        std.debug.print("  {s}\n", .{item});
    }
}

test "scope import lists match resolver surface" {
    try std.testing.expect(wasm_imports.Resolver.resolve("wasi_nn", "load_graph") != null);
    try std.testing.expect(wasm_imports.Resolver.resolve("wasi_nn", "get_output") != null);
    try std.testing.expect(wasm_imports.Resolver.resolve("zug_nn", "load_preloaded_graph") != null);
    try std.testing.expect(wasm_imports.Resolver.resolve("wasi_snapshot_preview1", "fd_write") != null);
    try std.testing.expect(wasm_imports.Resolver.resolve("wasi_snapshot_preview1", "random_get") != null);
}

test "scope includes active model compatibility surface" {
    try std.testing.expect(capabilities.isSupportedOperator("ai.onnx", "Conv"));
    try std.testing.expect(capabilities.isSupportedOperator("ai.onnx", "Gemm"));
    try std.testing.expect(capabilities.isSupportedOperator("ai.onnx", "Softmax"));
    try std.testing.expectEqual(@as(usize, 27), capabilities.supported_operator_names.len);
    try std.testing.expectEqual(@as(usize, 5), capabilities.supported_tensor_dtype_names.len);
}
