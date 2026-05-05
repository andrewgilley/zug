const std = @import("std");
const scope = @import("scope.zig");
const wasi_nn_abi = @import("wasi_nn_abi.zig");
const wasm_fixtures = @import("wasm/fixtures.zig");
const wasm_imports = @import("wasm/imports.zig");
const wasm_instance_mod = @import("wasm/instance.zig");
const wasm_interpreter = @import("wasm/interpreter.zig");
const wasm_manifest = @import("wasm/manifest.zig");
const wasm_module = @import("wasm/module.zig");
const wasm_runtime = @import("wasm/runtime.zig");

const wasm_page_size: usize = 64 * 1024;
const edge_initial_memory_bytes: usize = wasm_page_size;
const edge_max_memory_bytes: usize = 2 * wasm_page_size;
const edge_model_budget_bytes: usize = 64 * 1024;
const tiny_mnist_path = "models/tiny_mnist.onnx";

const DeploymentRunResult = struct {
    status: u32,
    pages_after_run: u32,
    output_shape_len: u32,
    output_byte_len: u32,
    stdout: []u8,
    output_bytes: []u8,

    fn deinit(self: *DeploymentRunResult, allocator: std.mem.Allocator) void {
        allocator.free(self.stdout);
        allocator.free(self.output_bytes);
        self.* = undefined;
    }
};

test "deployment profile runs constrained edge package under target memory budget" {
    const allocator = std.testing.allocator;

    const model_bytes = try readTinyMnistModel(allocator);
    defer allocator.free(model_bytes);

    var parsed = try parseEdgeModule(allocator, wasm_fixtures.constrained_edge_inference_flow);
    defer parsed.deinit(allocator);

    var manifest = try parseConstrainedEdgeManifest(allocator);
    defer manifest.deinit(allocator);

    try manifest.validate(&parsed, edgeCapabilities(), .{
        .export_name = "run",
        .has_preloaded_model = true,
        .model_len = model_bytes.len,
        .initial_memory_bytes = edge_initial_memory_bytes,
    });

    var result = try runConstrainedEdgePackage(allocator, model_bytes, edge_initial_memory_bytes);
    defer result.deinit(allocator);

    try std.testing.expectEqual(@intFromEnum(wasi_nn_abi.Status.ok), result.status);
    try std.testing.expectEqual(@as(u32, 2), result.pages_after_run);
    try std.testing.expect(result.pages_after_run * wasm_page_size <= edge_max_memory_bytes);
    try std.testing.expectEqualStrings("edge:start\nedge:done\n", result.stdout);
    try std.testing.expectEqual(@as(u32, 2), result.output_shape_len);
    try std.testing.expectEqual(@as(u32, 40), result.output_byte_len);
    try std.testing.expectEqual(@as(usize, 40), result.output_bytes.len);
}

test "deployment profile rejects package requests outside device budget" {
    const allocator = std.testing.allocator;

    var parsed = try parseEdgeModule(allocator, wasm_fixtures.constrained_edge_inference_flow);
    defer parsed.deinit(allocator);

    var manifest = try parseConstrainedEdgeManifest(allocator);
    defer manifest.deinit(allocator);

    try std.testing.expectError(error.ManifestMemoryTooLarge, manifest.validate(&parsed, edgeCapabilities(), .{
        .export_name = "run",
        .has_preloaded_model = true,
        .model_len = 32 * 1024,
        .initial_memory_bytes = 4 * wasm_page_size,
    }));

    try std.testing.expectError(error.ManifestModelTooLarge, manifest.validate(&parsed, edgeCapabilities(), .{
        .export_name = "run",
        .has_preloaded_model = true,
        .model_len = edge_model_budget_bytes + 1,
        .initial_memory_bytes = edge_initial_memory_bytes,
    }));

    try std.testing.expectError(error.ManifestRequiresModel, manifest.validate(&parsed, edgeCapabilities(), .{
        .export_name = "run",
        .has_preloaded_model = false,
        .model_len = 0,
        .initial_memory_bytes = edge_initial_memory_bytes,
    }));
}

test "deployment profile catches ABI drift between manifest and guest module" {
    const allocator = std.testing.allocator;

    var parsed = try parseEdgeModule(allocator, wasm_fixtures.wasi_nn_tiny_mnist_flow);
    defer parsed.deinit(allocator);

    var manifest = try parseConstrainedEdgeManifest(allocator);
    defer manifest.deinit(allocator);

    try std.testing.expectError(error.ManifestImportMissingFromModule, manifest.validate(&parsed, edgeCapabilities(), .{
        .export_name = "run",
        .has_preloaded_model = true,
        .model_len = 32 * 1024,
        .initial_memory_bytes = edge_initial_memory_bytes,
    }));
}

test "deployment cold restart produces deterministic edge output" {
    const allocator = std.testing.allocator;

    const model_bytes = try readTinyMnistModel(allocator);
    defer allocator.free(model_bytes);

    var first = try runConstrainedEdgePackage(allocator, model_bytes, edge_initial_memory_bytes);
    defer first.deinit(allocator);

    var second = try runConstrainedEdgePackage(allocator, model_bytes, edge_initial_memory_bytes);
    defer second.deinit(allocator);

    try std.testing.expectEqual(first.status, second.status);
    try std.testing.expectEqual(first.pages_after_run, second.pages_after_run);
    try std.testing.expectEqual(first.output_shape_len, second.output_shape_len);
    try std.testing.expectEqual(first.output_byte_len, second.output_byte_len);
    try std.testing.expectEqualSlices(u8, first.stdout, second.stdout);
    try std.testing.expectEqualSlices(u8, first.output_bytes, second.output_bytes);
}

test "deployment rejects malformed over the air model bytes without escaping sandbox" {
    const allocator = std.testing.allocator;
    const invalid_model_bytes = "not an onnx model";

    var result = try runConstrainedEdgePackage(allocator, invalid_model_bytes, edge_initial_memory_bytes);
    defer result.deinit(allocator);

    try std.testing.expect(result.status != @intFromEnum(wasi_nn_abi.Status.ok));
    try std.testing.expectEqual(@as(u32, 2), result.pages_after_run);
    try std.testing.expect(result.pages_after_run * wasm_page_size <= edge_max_memory_bytes);
    try std.testing.expectEqualStrings("edge:start\nedge:done\n", result.stdout);
}

fn readTinyMnistModel(allocator: std.mem.Allocator) ![]u8 {
    return try std.Io.Dir.cwd().readFileAlloc(
        std.Options.debug_io,
        tiny_mnist_path,
        allocator,
        .limited(scope.max_model_bytes),
    );
}

fn parseEdgeModule(
    allocator: std.mem.Allocator,
    bytes: []const u8,
) !wasm_module.Module {
    return wasm_runtime.Runtime.init(allocator).parseModule(bytes);
}

fn parseConstrainedEdgeManifest(allocator: std.mem.Allocator) !wasm_manifest.Manifest {
    return wasm_manifest.Manifest.parse(allocator,
        \\name = constrained-edge-inference
        \\runtime = zug-edge-runtime
        \\export = run
        \\requires_model = true
        \\model_encoding = onnx
        \\max_model_bytes = 64KiB
        \\min_memory_bytes = 64KiB
        \\max_memory_bytes = 128KiB
        \\requires_import = wasi_snapshot_preview1.fd_write
        \\requires_import = wasi_nn.load_graph
        \\requires_import = wasi_nn.init_execution_context
        \\requires_import = wasi_nn.set_input_by_index
        \\requires_import = wasi_nn.compute
        \\requires_import = wasi_nn.get_output_descriptor
        \\requires_import = wasi_nn.get_output
        \\
    );
}

fn edgeCapabilities() wasm_manifest.RuntimeCapabilities {
    return .{
        .max_model_bytes = edge_model_budget_bytes,
        .max_memory_bytes = edge_max_memory_bytes,
        .supports_preloaded_model = true,
    };
}

fn runConstrainedEdgePackage(
    allocator: std.mem.Allocator,
    model_bytes: []const u8,
    initial_memory_bytes: usize,
) !DeploymentRunResult {
    var parsed = try parseEdgeModule(allocator, wasm_fixtures.constrained_edge_inference_flow);
    defer parsed.deinit(allocator);

    var wasm_instance = try wasm_instance_mod.Instance.init(allocator, &parsed, initial_memory_bytes);
    defer wasm_instance.deinit();

    try wasm_instance.memory.write(1024, model_bytes);

    var host = wasi_nn_abi.Host.init(allocator);
    defer host.deinit();

    var surface = wasi_nn_abi.Surface.init(allocator, &host, &wasm_instance.memory);
    var resolver = wasm_imports.Resolver.init(&surface);
    defer resolver.deinit();
    try wasm_instance.bindImports(&resolver);

    var interpreter = wasm_interpreter.Interpreter.init(&wasm_instance);
    const result = (try interpreter.callExport("run", &.{
        .{ .i32 = std.math.cast(u32, model_bytes.len) orelse return error.ModelTooLarge },
    })) orelse return error.MissingReturnValue;

    const status = switch (result) {
        .i32 => |value| value,
        else => return error.ExpectedI32Status,
    };

    const pages_after_run = try wasm_instance.currentMemoryPages();
    const output_shape_len = try wasm_instance.memory.readU32(36);
    const output_byte_len = try wasm_instance.memory.readU32(40);
    const stdout = try allocator.dupe(u8, resolver.stdout.items);
    errdefer allocator.free(stdout);
    const output_bytes = try allocator.dupe(u8, try wasm_instance.memory.read(128, 40));
    errdefer allocator.free(output_bytes);

    return .{
        .status = status,
        .pages_after_run = pages_after_run,
        .output_shape_len = output_shape_len,
        .output_byte_len = output_byte_len,
        .stdout = stdout,
        .output_bytes = output_bytes,
    };
}
