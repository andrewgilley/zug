const std = @import("std");
const onnx = @import("proto/onnx.pb.zig");
const capabilities = @import("capabilities.zig");
const scope = @import("scope.zig");
const wasm_fixtures = @import("wasm/fixtures.zig");
const wasm_imports = @import("wasm/imports.zig");
const wasm_manifest = @import("wasm/manifest.zig");
const wasm_module = @import("wasm/module.zig");
const wasm_runtime = @import("wasm/runtime.zig");

const wasm_page_size: usize = 64 * 1024;

pub const ArtifactKind = enum {
    auto,
    onnx,
    wasm,
};

pub const Options = struct {
    artifact_path: []const u8,
    kind: ArtifactKind = .auto,
    manifest_path: ?[]const u8 = null,
    model_path: ?[]const u8 = null,
    export_name: ?[]const u8 = null,
    initial_memory_bytes: ?usize = null,
};

pub fn run(allocator: std.mem.Allocator, options: Options) !bool {
    const kind = try detectKind(options.artifact_path, options.kind);

    return switch (kind) {
        .onnx => try checkOnnxPath(allocator, options.artifact_path),
        .wasm => try checkWasmPath(allocator, options),
        .auto => unreachable,
    };
}

pub fn parseByteCount(value: []const u8) !usize {
    const suffixes = [_]struct {
        suffix: []const u8,
        multiplier: usize,
    }{
        .{ .suffix = "KiB", .multiplier = 1024 },
        .{ .suffix = "MiB", .multiplier = 1024 * 1024 },
        .{ .suffix = "GiB", .multiplier = 1024 * 1024 * 1024 },
        .{ .suffix = "KB", .multiplier = 1000 },
        .{ .suffix = "MB", .multiplier = 1000 * 1000 },
        .{ .suffix = "GB", .multiplier = 1000 * 1000 * 1000 },
    };

    for (suffixes) |suffix| {
        if (std.mem.endsWith(u8, value, suffix.suffix)) {
            const number = try std.fmt.parseInt(usize, trim(value[0 .. value.len - suffix.suffix.len]), 10);
            return std.math.mul(usize, number, suffix.multiplier);
        }
    }

    return std.fmt.parseInt(usize, value, 10);
}

fn checkOnnxPath(allocator: std.mem.Allocator, path: []const u8) !bool {
    const bytes = try std.Io.Dir.cwd().readFileAlloc(
        std.Options.debug_io,
        path,
        allocator,
        .limited(scope.max_model_bytes),
    );
    defer allocator.free(bytes);

    return checkOnnxBytes(allocator, path, bytes);
}

fn checkOnnxBytes(allocator: std.mem.Allocator, label: []const u8, bytes: []const u8) !bool {
    std.debug.print("zug check onnx\n", .{});
    std.debug.print("  file: {s}\n", .{label});
    std.debug.print("  bytes: {d}\n", .{bytes.len});

    var reader: std.Io.Reader = .fixed(bytes);
    var model = onnx.ModelProto.decode(&reader, allocator) catch |err| {
        std.debug.print("  decode: fail ({s})\n", .{@errorName(err)});
        std.debug.print("  status: fail\n", .{});
        return false;
    };
    defer model.deinit(allocator);

    std.debug.print("  decode: pass\n", .{});

    var report = try capabilities.analyze(allocator, &model);
    defer report.deinit(allocator);

    const ok = onnxReportPassesCheck(report);
    printOnnxReport(report);
    std.debug.print("  status: {s}\n", .{if (ok) "pass" else "fail"});
    return ok;
}

fn printOnnxReport(report: capabilities.Report) void {
    std.debug.print("  opsets: {d}\n", .{report.opsets.items.len});
    for (report.opsets.items) |opset| {
        std.debug.print("    {s}: ", .{opset.domain});
        if (opset.version) |version| {
            std.debug.print("{d}\n", .{version});
        } else {
            std.debug.print("<missing>\n", .{});
        }
    }

    std.debug.print("  operators: {d}\n", .{report.operators.items.len});
    for (report.operators.items) |operator| {
        std.debug.print("    {s}::{s} x{d} {s}\n", .{
            operator.domain,
            operator.op_type,
            operator.count,
            if (operator.supported) "supported" else "unsupported",
        });
    }

    std.debug.print("  issues: {d}\n", .{report.issues.items.len});
    for (report.issues.items) |issue| {
        std.debug.print("    {s} {s}: {s}", .{
            onnxIssueSeverity(issue.kind),
            @tagName(issue.kind),
            issue.subject,
        });
        if (issue.detail.len != 0) {
            std.debug.print(" - {s}", .{issue.detail});
        }
        std.debug.print("\n", .{});
    }
}

fn onnxReportPassesCheck(report: capabilities.Report) bool {
    for (report.issues.items) |issue| {
        if (onnxIssueIsFatal(issue.kind)) return false;
    }

    return true;
}

fn onnxIssueSeverity(kind: capabilities.IssueKind) []const u8 {
    return if (onnxIssueIsFatal(kind)) "error" else "warning";
}

fn onnxIssueIsFatal(kind: capabilities.IssueKind) bool {
    return switch (kind) {
        .dynamic_dimension => false,
        else => true,
    };
}

fn checkWasmPath(allocator: std.mem.Allocator, options: Options) !bool {
    const bytes = try std.Io.Dir.cwd().readFileAlloc(
        std.Options.debug_io,
        options.artifact_path,
        allocator,
        .limited(scope.max_wasm_bytes),
    );
    defer allocator.free(bytes);

    var ok = try checkWasmBytes(allocator, options, bytes);

    if (options.model_path) |model_path| {
        std.debug.print("\n", .{});
        ok = (try checkOnnxPath(allocator, model_path)) and ok;
    }

    return ok;
}

fn checkWasmBytes(allocator: std.mem.Allocator, options: Options, bytes: []const u8) !bool {
    std.debug.print("zug check wasm\n", .{});
    std.debug.print("  file: {s}\n", .{options.artifact_path});
    std.debug.print("  bytes: {d}\n", .{bytes.len});

    const runtime = wasm_runtime.Runtime.init(allocator);
    var parsed = runtime.parseModule(bytes) catch |err| {
        std.debug.print("  parse: fail ({s})\n", .{@errorName(err)});
        std.debug.print("  status: fail\n", .{});
        return false;
    };
    defer parsed.deinit(allocator);

    std.debug.print("  parse: pass\n", .{});

    var ok = true;
    if (runtime.validateModule(&parsed)) {
        std.debug.print("  validate: pass\n", .{});
    } else |err| {
        std.debug.print("  validate: fail ({s})\n", .{@errorName(err)});
        ok = false;
    }

    const imports_ok = printImportReport(&parsed);
    const resources_ok = printWasmResourceReport(&parsed, options.initial_memory_bytes);
    const exports_ok = printExportReport(&parsed, options.export_name);
    ok = ok and imports_ok and resources_ok and exports_ok;

    if (options.manifest_path) |manifest_path| {
        const manifest_ok = try checkManifest(allocator, &parsed, options, manifest_path);
        ok = ok and manifest_ok;
    }

    std.debug.print("  status: {s}\n", .{if (ok) "pass" else "fail"});
    return ok;
}

fn printImportReport(parsed: *const wasm_module.Module) bool {
    var function_imports: usize = 0;
    var unsupported_function_imports: usize = 0;
    var unsupported_non_function_imports: usize = 0;

    std.debug.print("  imports: {d}\n", .{parsed.imports.items.len});
    for (parsed.imports.items) |import| {
        switch (import.kind) {
            .function => {
                function_imports += 1;
                const supported = wasm_imports.Resolver.resolve(import.module, import.name) != null;
                if (!supported) {
                    unsupported_function_imports += 1;
                }
                std.debug.print("    func {s}.{s} {s}\n", .{
                    import.module,
                    import.name,
                    if (supported) "supported" else "unsupported",
                });
            },
            .memory, .table, .global => {
                unsupported_non_function_imports += 1;
                std.debug.print("    {s} {s}.{s} unsupported\n", .{
                    @tagName(import.kind),
                    import.module,
                    import.name,
                });
            },
        }
    }

    std.debug.print("  function_imports: {d}\n", .{function_imports});
    std.debug.print("  unsupported_function_imports: {d}\n", .{unsupported_function_imports});
    std.debug.print("  unsupported_non_function_imports: {d}\n", .{unsupported_non_function_imports});
    return unsupported_function_imports == 0 and unsupported_non_function_imports == 0;
}

fn importsSupported(parsed: *const wasm_module.Module) bool {
    for (parsed.imports.items) |import| {
        switch (import.kind) {
            .function => {
                if (wasm_imports.Resolver.resolve(import.module, import.name) == null) return false;
            },
            .memory, .table, .global => return false,
        }
    }

    return true;
}

fn printWasmResourceReport(parsed: *const wasm_module.Module, requested_memory_bytes: ?usize) bool {
    var ok = true;
    std.debug.print("  memories: {d}\n", .{parsed.memories.items.len});

    if (parsed.memories.items.len > 1) ok = false;

    const min_memory = minimumMemoryBytes(parsed) catch |err| {
        std.debug.print("  memory_min_bytes: fail ({s})\n", .{@errorName(err)});
        return false;
    };
    const requested = requested_memory_bytes orelse @max(scope.default_wasm_memory_bytes, min_memory);

    if (requested > scope.max_wasm_memory_bytes) ok = false;
    if (requested < min_memory) ok = false;

    std.debug.print("  memory_min_bytes: {d}\n", .{min_memory});
    std.debug.print("  requested_initial_memory_bytes: {d}\n", .{requested});
    std.debug.print("  max_runtime_memory_bytes: {d}\n", .{scope.max_wasm_memory_bytes});

    for (parsed.memories.items, 0..) |memory, index| {
        std.debug.print("    memory[{d}] min_pages={d}", .{ index, memory.limits.min });
        if (memory.limits.max) |max| {
            std.debug.print(" max_pages={d}", .{max});
        } else {
            std.debug.print(" max_pages=<runtime default>", .{});
        }
        std.debug.print("\n", .{});
    }

    return ok;
}

fn printExportReport(parsed: *const wasm_module.Module, export_name: ?[]const u8) bool {
    var has_requested_export = export_name == null;
    std.debug.print("  exports: {d}\n", .{parsed.exports.items.len});

    for (parsed.exports.items) |exported| {
        if (export_name) |requested| {
            if (exported.kind == .function and std.mem.eql(u8, exported.name, requested)) {
                has_requested_export = true;
            }
        }

        std.debug.print("    {s} {s} index={d}\n", .{
            @tagName(exported.kind),
            exported.name,
            exported.index,
        });
    }

    if (export_name) |requested| {
        std.debug.print("  requested_export: {s} {s}\n", .{
            requested,
            if (has_requested_export) "found" else "missing",
        });
    }

    return has_requested_export;
}

fn checkManifest(
    allocator: std.mem.Allocator,
    parsed: *const wasm_module.Module,
    options: Options,
    manifest_path: []const u8,
) !bool {
    const manifest_bytes = try std.Io.Dir.cwd().readFileAlloc(
        std.Options.debug_io,
        manifest_path,
        allocator,
        .limited(wasm_manifest.max_manifest_bytes),
    );
    defer allocator.free(manifest_bytes);

    var manifest = wasm_manifest.Manifest.parse(allocator, manifest_bytes) catch |err| {
        std.debug.print("  manifest: fail ({s})\n", .{@errorName(err)});
        return false;
    };
    defer manifest.deinit(allocator);

    const model_len = if (options.model_path) |model_path| blk: {
        const model_bytes = try std.Io.Dir.cwd().readFileAlloc(
            std.Options.debug_io,
            model_path,
            allocator,
            .limited(scope.max_model_bytes),
        );
        defer allocator.free(model_bytes);
        break :blk model_bytes.len;
    } else 0;

    const initial_memory = options.initial_memory_bytes orelse @max(
        scope.default_wasm_memory_bytes,
        try minimumMemoryBytes(parsed),
    );
    const export_name = options.export_name orelse manifest.export_name orelse "run";

    manifest.validate(parsed, .{
        .max_model_bytes = scope.max_model_bytes,
        .max_memory_bytes = scope.max_wasm_memory_bytes,
        .supports_preloaded_model = true,
    }, .{
        .export_name = export_name,
        .has_preloaded_model = options.model_path != null,
        .model_len = model_len,
        .initial_memory_bytes = initial_memory,
    }) catch |err| {
        std.debug.print("  manifest: fail ({s})\n", .{@errorName(err)});
        return false;
    };

    std.debug.print("  manifest: pass\n", .{});
    return true;
}

fn minimumMemoryBytes(parsed: *const wasm_module.Module) !usize {
    if (parsed.memories.items.len == 0) return 0;
    if (parsed.memories.items.len > 1) return error.MultipleMemoriesUnsupported;

    const pages = std.math.cast(usize, parsed.memories.items[0].limits.min) orelse {
        return error.MemorySizeTooLarge;
    };

    return try std.math.mul(usize, pages, wasm_page_size);
}

fn detectKind(path: []const u8, requested: ArtifactKind) !ArtifactKind {
    if (requested != .auto) return requested;
    if (endsWithIgnoreCase(path, ".onnx")) return .onnx;
    if (endsWithIgnoreCase(path, ".wasm")) return .wasm;
    return error.UnknownCheckArtifactKind;
}

fn endsWithIgnoreCase(value: []const u8, suffix: []const u8) bool {
    if (value.len < suffix.len) return false;
    const tail = value[value.len - suffix.len ..];
    return std.ascii.eqlIgnoreCase(tail, suffix);
}

fn trim(value: []const u8) []const u8 {
    return std.mem.trim(u8, value, " \t\r");
}

test "check accepts supported wasm imports" {
    const allocator = std.testing.allocator;

    var parsed = try wasm_module.Module.parse(allocator, wasm_fixtures.return_i32_seven);
    defer parsed.deinit(allocator);

    try wasm_runtime.Runtime.init(allocator).validateModule(&parsed);
    try std.testing.expect(importsSupported(&parsed));
}

test "check rejects unsupported wasm import" {
    const allocator = std.testing.allocator;

    const bytes =
        "\x00asm\x01\x00\x00\x00" ++
        "\x01\x04" ++
        "\x01" ++
        "\x60\x00\x00" ++
        "\x02\x0f" ++
        "\x01" ++
        "\x03env" ++
        "\x07missing" ++
        "\x00\x00";

    var parsed = try wasm_module.Module.parse(allocator, bytes);
    defer parsed.deinit(allocator);

    try wasm_runtime.Runtime.init(allocator).validateModule(&parsed);
    try std.testing.expect(!importsSupported(&parsed));
}

test "parseByteCount supports binary suffixes" {
    try std.testing.expectEqual(@as(usize, 64 * 1024), try parseByteCount("64KiB"));
    try std.testing.expectEqual(@as(usize, 2 * 1024 * 1024), try parseByteCount("2MiB"));
    try std.testing.expectEqual(@as(usize, 512), try parseByteCount("512"));
}
