const std = @import("std");
const accelerator = @import("accelerator.zig");
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
    "wasi_snapshot_preview1.fd_close",
    "wasi_snapshot_preview1.fd_fdstat_get",
    "wasi_snapshot_preview1.fd_filestat_get",
    "wasi_snapshot_preview1.fd_prestat_get",
    "wasi_snapshot_preview1.fd_prestat_dir_name",
    "wasi_snapshot_preview1.fd_read",
    "wasi_snapshot_preview1.fd_readdir",
    "wasi_snapshot_preview1.fd_seek",
    "wasi_snapshot_preview1.fd_write",
    "wasi_snapshot_preview1.path_filestat_get",
    "wasi_snapshot_preview1.path_open",
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

pub const zug_gpu_scope = [_][]const u8{
    "zug_gpu.device_count",
    "zug_gpu.device_kind",
    "zug_gpu.device_memory",
    "zug_gpu.device_queue_count",
    "zug_gpu.select_device",
    "zug_gpu.selected_device",
    "zug_gpu.open_device",
    "zug_gpu.default_queue",
    "zug_gpu.create_buffer",
    "zug_gpu.write_buffer",
    "zug_gpu.read_buffer",
    "zug_gpu.dispatch_compute_stub",
};

pub const model_execution_scope = [_][]const u8{
    "ONNX graph decoding through generated protobuf bindings",
    "CPU execution target",
    "accelerator backend catalog for CUDA, TensorRT, ROCm, Vulkan, Metal, DirectML, OpenVINO, CoreML, NNAPI, WebGPU and Edge TPU",
    "f32 inference path for WASI-NN guest inputs",
    "direct host ONNX execution through Session and Executor",
    "raw tensor file input and raw output/expectation checks",
    "runtime benchmark reporting for timing and allocation pressure",
};

pub const telemetry_scope = [_][]const u8{
    "agent GET /telemetry",
    "agent GET /telemetry/events",
    "agent GET /telemetry/metrics",
    "agent GET /telemetry/traces",
    "agent POST /telemetry/events",
    "agent POST /telemetry/metrics",
    "agent POST /telemetry/traces",
    "OTLP/HTTP JSON ingest POST /v1/logs",
    "OTLP/HTTP JSON ingest POST /v1/metrics",
    "OTLP/HTTP JSON ingest POST /v1/traces",
    "OTLP JSON export GET /telemetry/otlp/logs",
    "OTLP JSON export GET /telemetry/otlp/metrics",
    "OTLP JSON export GET /telemetry/otlp/traces",
    "in-memory event, scalar metric and trace span store",
};

pub const next_capability_frontiers = [_][]const u8{
    "scratch-buffer reuse for steady-state model execution",
    "specialized Conv kernels for MobileNet-style depthwise and pointwise convolutions",
    "compiled ONNX execution plan with numeric tensor slots instead of per-node string lookup",
    "broader WASI Preview 1 write policy and guest HTTP capability mapping",
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
    printAcceleratorBackends();
    printList("wasm core", &wasm_core_scope);
    printList("wasi preview1 imports", &wasi_preview1_scope);
    printList("wasi-nn imports", &wasi_nn_scope);
    printList("zug host extension imports", &zug_nn_scope);
    printList("zug gpu extension imports", &zug_gpu_scope);
    printList("telemetry", &telemetry_scope);
    printList("next capability frontiers", &next_capability_frontiers);
}

fn printAcceleratorBackends() void {
    const caps = accelerator.defaultCapabilities();
    std.debug.print("accelerator backends: {d}\n", .{caps.backends.len});
    for (caps.backends) |backend| {
        std.debug.print("  {s}: {s}\n", .{
            accelerator.backendKindName(backend.kind),
            accelerator.backendStatusName(backend.status),
        });
    }
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
    try std.testing.expect(wasm_imports.Resolver.resolve("wasi_snapshot_preview1", "path_open") != null);
    try std.testing.expect(wasm_imports.Resolver.resolve("wasi_snapshot_preview1", "fd_read") != null);
    try std.testing.expect(wasm_imports.Resolver.resolve("wasi_snapshot_preview1", "fd_readdir") != null);
    try std.testing.expect(wasm_imports.Resolver.resolve("zug_gpu", "device_count") != null);
    try std.testing.expect(wasm_imports.Resolver.resolve("zug_gpu", "device_memory") != null);
    try std.testing.expect(wasm_imports.Resolver.resolve("zug_gpu", "device_queue_count") != null);
    try std.testing.expect(wasm_imports.Resolver.resolve("zug_gpu", "open_device") != null);
    try std.testing.expect(wasm_imports.Resolver.resolve("zug_gpu", "create_buffer") != null);
    try std.testing.expect(wasm_imports.Resolver.resolve("zug_gpu", "dispatch_compute_stub") != null);
}

test "scope includes active model compatibility surface" {
    try std.testing.expect(capabilities.isSupportedOperator("ai.onnx", "Conv"));
    try std.testing.expect(capabilities.isSupportedOperator("ai.onnx", "Gemm"));
    try std.testing.expect(capabilities.isSupportedOperator("ai.onnx", "Softmax"));
    try std.testing.expectEqual(@as(usize, 45), capabilities.supported_operator_names.len);
    try std.testing.expectEqual(@as(usize, 5), capabilities.supported_tensor_dtype_names.len);
    try std.testing.expect(accelerator.defaultCapabilities().supportsGraphExecution(.cpu));
    try std.testing.expect(!accelerator.defaultCapabilities().supportsGraphExecution(.cuda));
}
