const std = @import("std");
const accelerator = @import("accelerator.zig");
const capabilities = @import("capabilities.zig");
const gpu = @import("gpu.zig");
const network = @import("network.zig");
const onnx = @import("proto/onnx.pb.zig");
const scope = @import("scope.zig");
const target_profile = @import("target.zig");
const wasi_nn_abi = @import("wasi_nn_abi.zig");
const workload = @import("workload.zig");
const wasm_compatibility = @import("wasm/compatibility.zig");
const wasm_imports = @import("wasm/imports.zig");
const wasm_interpreter = @import("wasm/interpreter.zig");
const wasm_runtime = @import("wasm/runtime.zig");

const wasm_page_size: usize = 64 * 1024;

pub const Options = struct {
    path: []const u8,
    profile_name: ?[]const u8 = null,
    args: []const u32 = &.{},
    stdin: []const u8 = &.{},
    gpu: gpu.Capabilities = .{},
    accelerators: accelerator.Capabilities = accelerator.defaultCapabilities(),
};

pub const Result = struct {
    path: []const u8,
    name: []const u8,
    profile: []const u8,
    entrypoint: []const u8,
    wasm_path: []const u8,
    model_path: []const u8,
    wasm_bytes: usize,
    model_bytes: usize,
    memory_bytes: usize,
    duration_ns: u64,
    stdout: []const u8,
    stderr: []const u8,
    exit_code: ?u32,
    value: ?wasm_interpreter.Value,

    pub fn deinit(self: *Result, allocator: std.mem.Allocator) void {
        allocator.free(self.path);
        allocator.free(self.name);
        allocator.free(self.profile);
        allocator.free(self.entrypoint);
        allocator.free(self.wasm_path);
        allocator.free(self.model_path);
        allocator.free(self.stdout);
        allocator.free(self.stderr);
        self.* = undefined;
    }
};

pub fn run(allocator: std.mem.Allocator, options: Options) !Result {
    const started_ns = monotonicNowNs();

    var loaded = try workload.Loaded.load(allocator, options.path);
    defer loaded.deinit(allocator);

    const requested_profile_name = options.profile_name orelse loaded.manifest.targetName();
    const profile = try target_profile.resolve(requested_profile_name);

    try loaded.manifest.validateForTarget(profile);

    var network_report = try network.analyze(allocator, loaded.manifest.network, profile.network);
    defer network_report.deinit(allocator);
    if (!network_report.supported()) return error.WorkloadNetworkUnsupported;

    const wasm_bytes = try std.Io.Dir.cwd().readFileAlloc(
        std.Options.debug_io,
        loaded.wasm_path,
        allocator,
        .limited(scope.max_wasm_bytes),
    );
    defer allocator.free(wasm_bytes);

    const model_bytes = if (loaded.model_path) |model_path|
        try std.Io.Dir.cwd().readFileAlloc(
            std.Options.debug_io,
            model_path,
            allocator,
            .limited(scope.max_model_bytes),
        )
    else
        null;
    defer if (model_bytes) |bytes| allocator.free(bytes);

    if (model_bytes) |bytes| {
        if (bytes.len > profile.max_model_bytes) return error.WorkloadModelTooLarge;
        try validateOnnxModelBytes(allocator, bytes);
    }

    const initial_memory_bytes = try workloadInitialMemoryBytes(loaded, profile);
    const runtime = wasm_runtime.Runtime.init(allocator);

    var parsed = try runtime.parseModule(wasm_bytes);
    defer parsed.deinit(allocator);

    var compatibility_report = try wasm_compatibility.analyzeModule(allocator, wasm_bytes, &parsed, compatibilityTarget(profile), .{
        .initial_memory_bytes = initial_memory_bytes,
        .export_name = loaded.manifest.entrypointName(),
    });
    defer compatibility_report.deinit(allocator);

    if (!compatibility_report.supported()) return error.WorkloadWasmUnsupported;
    if (loaded.manifest.requires.wasi_nn and !hasMlImport(compatibility_report)) return error.WorkloadMissingMlImport;

    var wasm_instance = try runtime.instantiate(&parsed, initial_memory_bytes);
    defer wasm_instance.deinit();

    var host = wasi_nn_abi.Host.initWithAccelerators(allocator, options.accelerators);
    defer host.deinit();

    var surface = wasi_nn_abi.Surface.init(allocator, &host, &wasm_instance.memory);
    surface.setPreloadedModel(model_bytes);

    var resolver = wasm_imports.Resolver.init(&surface);
    resolver.stdin = options.stdin;
    resolver.gpu = options.gpu;
    defer resolver.deinit();

    try wasm_instance.bindImports(&resolver);

    var interpreter = wasm_interpreter.Interpreter.init(&wasm_instance);
    try interpreter.runStart();

    var args: std.ArrayList(wasm_interpreter.Value) = .empty;
    defer args.deinit(allocator);

    for (options.args) |arg| {
        try args.append(allocator, .{ .i32 = arg });
    }

    const value = try interpreter.callExport(loaded.manifest.entrypointName(), args.items);

    const owned_stdout = try allocator.dupe(u8, resolver.stdout.items);
    errdefer allocator.free(owned_stdout);

    const owned_stderr = try allocator.dupe(u8, resolver.stderr.items);
    errdefer allocator.free(owned_stderr);

    const finished_ns = monotonicNowNs();

    const owned_path = try allocator.dupe(u8, loaded.root_path);
    errdefer allocator.free(owned_path);

    const owned_name = try allocator.dupe(u8, loaded.manifest.name orelse loaded.root_path);
    errdefer allocator.free(owned_name);

    const owned_profile = try allocator.dupe(u8, profile.name);
    errdefer allocator.free(owned_profile);

    const owned_entrypoint = try allocator.dupe(u8, loaded.manifest.entrypointName());
    errdefer allocator.free(owned_entrypoint);

    const owned_wasm_path = try allocator.dupe(u8, loaded.wasm_path);
    errdefer allocator.free(owned_wasm_path);

    const owned_model_path = try allocator.dupe(u8, loaded.model_path orelse "");
    errdefer allocator.free(owned_model_path);

    return .{
        .path = owned_path,
        .name = owned_name,
        .profile = owned_profile,
        .entrypoint = owned_entrypoint,
        .wasm_path = owned_wasm_path,
        .model_path = owned_model_path,
        .wasm_bytes = wasm_bytes.len,
        .model_bytes = if (model_bytes) |bytes| bytes.len else 0,
        .memory_bytes = wasm_instance.memory_bytes.len,
        .duration_ns = finished_ns -| started_ns,
        .stdout = owned_stdout,
        .stderr = owned_stderr,
        .exit_code = resolver.exit_code,
        .value = value,
    };
}

fn validateOnnxModelBytes(allocator: std.mem.Allocator, bytes: []const u8) !void {
    var reader: std.Io.Reader = .fixed(bytes);
    var model = try onnx.ModelProto.decode(&reader, allocator);
    defer model.deinit(allocator);

    var report = try capabilities.analyze(allocator, &model);
    defer report.deinit(allocator);

    for (report.issues.items) |issue| {
        if (onnxIssueIsFatal(issue.kind)) return error.WorkloadModelUnsupported;
    }
}

fn onnxIssueIsFatal(kind: capabilities.IssueKind) bool {
    return switch (kind) {
        .dynamic_dimension => false,
        else => true,
    };
}

fn compatibilityTarget(profile: target_profile.Profile) wasm_compatibility.Target {
    return .{
        .default_memory_bytes = profile.default_memory_bytes,
        .max_memory_bytes = profile.max_memory_bytes,
    };
}

fn workloadInitialMemoryBytes(loaded: workload.Loaded, profile: target_profile.Profile) !usize {
    const requested = loaded.manifest.requires.memory_bytes orelse profile.default_memory_bytes;
    const aligned = try alignToWasmPage(requested);
    if (aligned > profile.max_memory_bytes) return error.WorkloadMemoryTooLarge;
    return aligned;
}

fn alignToWasmPage(value: usize) !usize {
    const plus_page = try std.math.add(usize, value, wasm_page_size - 1);
    return plus_page - (plus_page % wasm_page_size);
}

fn hasMlImport(report: wasm_compatibility.Report) bool {
    for (report.imports.details.items) |import| {
        if (std.mem.eql(u8, import.module_name, "wasi_nn") or
            std.mem.eql(u8, import.module_name, "zug_nn"))
        {
            return true;
        }
    }

    return false;
}

fn monotonicNowNs() u64 {
    const value = std.Io.Clock.awake.now(std.Options.debug_io).toNanoseconds();
    return std.math.cast(u64, value) orelse 0;
}

test "workload runner aligns requested memory to wasm pages" {
    try std.testing.expectEqual(@as(usize, 64 * 1024), try alignToWasmPage(1));
    try std.testing.expectEqual(@as(usize, 2 * 1024 * 1024), try alignToWasmPage(2 * 1024 * 1024));
}
