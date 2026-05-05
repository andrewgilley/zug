const std = @import("std");
const onnx = @import("proto/onnx.pb.zig");
const capabilities = @import("capabilities.zig");
const scope = @import("scope.zig");
const wasm_compatibility = @import("wasm/compatibility.zig");
const wasm_manifest = @import("wasm/manifest.zig");
const wasm_module = @import("wasm/module.zig");

const wasm_page_size: usize = 64 * 1024;

pub const ArtifactKind = enum {
    auto,
    onnx,
    wasm,
};

pub const OutputFormat = enum {
    text,
    json,
};

pub const Options = struct {
    artifact_path: []const u8,
    kind: ArtifactKind = .auto,
    manifest_path: ?[]const u8 = null,
    model_path: ?[]const u8 = null,
    export_name: ?[]const u8 = null,
    initial_memory_bytes: ?usize = null,
    output_format: OutputFormat = .text,
};

const ManifestStatus = struct {
    path: []const u8,
    error_name: ?[]const u8 = null,

    fn supported(self: ManifestStatus) bool {
        return self.error_name == null;
    }
};

pub fn run(allocator: std.mem.Allocator, options: Options) !bool {
    const kind = try detectKind(options.artifact_path, options.kind);

    return switch (kind) {
        .onnx => try checkOnnxPath(allocator, options.artifact_path, options.output_format),
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

fn checkOnnxPath(allocator: std.mem.Allocator, path: []const u8, output_format: OutputFormat) !bool {
    const bytes = try std.Io.Dir.cwd().readFileAlloc(
        std.Options.debug_io,
        path,
        allocator,
        .limited(scope.max_model_bytes),
    );
    defer allocator.free(bytes);

    return checkOnnxBytes(allocator, path, bytes, output_format);
}

fn checkOnnxBytes(allocator: std.mem.Allocator, label: []const u8, bytes: []const u8, output_format: OutputFormat) !bool {
    if (output_format == .text) {
        std.debug.print("zug check onnx\n", .{});
        std.debug.print("  file: {s}\n", .{label});
        std.debug.print("  bytes: {d}\n", .{bytes.len});
    }

    var reader: std.Io.Reader = .fixed(bytes);
    var model = onnx.ModelProto.decode(&reader, allocator) catch |err| {
        if (output_format == .json) {
            printOnnxDecodeFailureJson(label, bytes.len, err);
        } else {
            std.debug.print("  decode: fail ({s})\n", .{@errorName(err)});
            std.debug.print("  status: fail\n", .{});
        }
        return false;
    };
    defer model.deinit(allocator);

    if (output_format == .text) {
        std.debug.print("  decode: pass\n", .{});
    }

    var report = try capabilities.analyze(allocator, &model);
    defer report.deinit(allocator);

    const ok = onnxReportPassesCheck(report);
    if (output_format == .json) {
        printOnnxCheckJson(label, bytes.len, report, ok);
    } else {
        printOnnxReport(report);
        std.debug.print("  status: {s}\n", .{if (ok) "pass" else "fail"});
    }
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

fn printOnnxDecodeFailureJson(label: []const u8, byte_len: usize, err: anyerror) void {
    std.debug.print("{{\"kind\":\"onnx\",\"file\":", .{});
    printJsonString(label);
    std.debug.print(",\"bytes\":{d},\"decode\":{{\"status\":\"fail\",\"error\":", .{byte_len});
    printJsonString(@errorName(err));
    std.debug.print("}},\"status\":\"fail\"}}\n", .{});
}

fn printOnnxCheckJson(label: []const u8, byte_len: usize, report: capabilities.Report, ok: bool) void {
    std.debug.print("{{\"kind\":\"onnx\",\"file\":", .{});
    printJsonString(label);
    std.debug.print(",\"bytes\":{d},\"decode\":{{\"status\":\"pass\"}},\"opsets\":[", .{byte_len});

    for (report.opsets.items, 0..) |opset, index| {
        if (index != 0) std.debug.print(",", .{});
        std.debug.print("{{\"domain\":", .{});
        printJsonString(opset.domain);
        std.debug.print(",\"version\":", .{});
        if (opset.version) |version| {
            std.debug.print("{d}", .{version});
        } else {
            std.debug.print("null", .{});
        }
        std.debug.print("}}", .{});
    }

    std.debug.print("],\"operators\":[", .{});
    for (report.operators.items, 0..) |operator, index| {
        if (index != 0) std.debug.print(",", .{});
        std.debug.print("{{\"domain\":", .{});
        printJsonString(operator.domain);
        std.debug.print(",\"op_type\":", .{});
        printJsonString(operator.op_type);
        std.debug.print(",\"count\":{d},\"supported\":{}}}", .{ operator.count, operator.supported });
    }

    std.debug.print("],\"issues\":[", .{});
    for (report.issues.items, 0..) |issue, index| {
        if (index != 0) std.debug.print(",", .{});
        std.debug.print("{{\"severity\":", .{});
        printJsonString(onnxIssueSeverity(issue.kind));
        std.debug.print(",\"kind\":", .{});
        printJsonString(@tagName(issue.kind));
        std.debug.print(",\"subject\":", .{});
        printJsonString(issue.subject);
        std.debug.print(",\"detail\":", .{});
        printJsonString(issue.detail);
        std.debug.print("}}", .{});
    }

    std.debug.print("],\"status\":\"{s}\"}}\n", .{if (ok) "pass" else "fail"});
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
        ok = (try checkOnnxPath(allocator, model_path, options.output_format)) and ok;
    }

    return ok;
}

fn checkWasmBytes(allocator: std.mem.Allocator, options: Options, bytes: []const u8) !bool {
    if (options.output_format == .text) {
        std.debug.print("zug check wasm\n", .{});
        std.debug.print("  file: {s}\n", .{options.artifact_path});
        std.debug.print("  bytes: {d}\n", .{bytes.len});
    }

    var report = try wasm_compatibility.analyzeBytes(allocator, bytes, .{}, .{
        .initial_memory_bytes = options.initial_memory_bytes,
        .export_name = options.export_name,
    });
    defer report.deinit(allocator);

    var ok = report.supported();
    var manifest_status: ?ManifestStatus = null;

    if (options.manifest_path) |manifest_path| {
        if (report.parse_error) |_| {
            ok = false;
            manifest_status = .{
                .path = manifest_path,
                .error_name = "WasmParseFailed",
            };
        } else {
            var parsed = try wasm_module.Module.parse(allocator, bytes);
            defer parsed.deinit(allocator);

            manifest_status = try checkManifest(allocator, &parsed, options, manifest_path, options.output_format);
            ok = ok and manifest_status.?.supported();
        }
    }

    if (options.output_format == .json) {
        printWasmCheckJson(options.artifact_path, bytes.len, report, manifest_status, ok);
    } else {
        report.print();
        if (manifest_status) |manifest| printManifestStatusText(manifest);
        std.debug.print("  status: {s}\n", .{if (ok) "pass" else "fail"});
    }
    return ok;
}

fn printWasmCheckJson(
    artifact_path: []const u8,
    byte_len: usize,
    report: wasm_compatibility.Report,
    manifest_status: ?ManifestStatus,
    ok: bool,
) void {
    std.debug.print("{{\"kind\":\"wasm\",\"file\":", .{});
    printJsonString(artifact_path);
    std.debug.print(",\"bytes\":{d},\"status\":\"{s}\",\"compatibility\":", .{
        byte_len,
        if (ok) "pass" else "fail",
    });
    report.printJson();

    if (manifest_status) |manifest| {
        std.debug.print(",\"manifest\":", .{});
        printManifestStatusJson(manifest);
    }

    std.debug.print("}}\n", .{});
}

fn printManifestStatusText(manifest: ManifestStatus) void {
    if (manifest.error_name) |error_name| {
        std.debug.print("  manifest: fail ({s})\n", .{error_name});
    } else {
        std.debug.print("  manifest: pass\n", .{});
    }
}

fn printManifestStatusJson(manifest: ManifestStatus) void {
    std.debug.print("{{\"path\":", .{});
    printJsonString(manifest.path);
    std.debug.print(",\"status\":\"{s}\",\"error\":", .{if (manifest.supported()) "pass" else "fail"});
    if (manifest.error_name) |error_name| {
        printJsonString(error_name);
    } else {
        std.debug.print("null", .{});
    }
    std.debug.print("}}", .{});
}

fn checkManifest(
    allocator: std.mem.Allocator,
    parsed: *const wasm_module.Module,
    options: Options,
    manifest_path: []const u8,
    output_format: OutputFormat,
) !ManifestStatus {
    _ = output_format;

    const manifest_bytes = try std.Io.Dir.cwd().readFileAlloc(
        std.Options.debug_io,
        manifest_path,
        allocator,
        .limited(wasm_manifest.max_manifest_bytes),
    );
    defer allocator.free(manifest_bytes);

    var manifest = wasm_manifest.Manifest.parse(allocator, manifest_bytes) catch |err| {
        return .{
            .path = manifest_path,
            .error_name = @errorName(err),
        };
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
        return .{
            .path = manifest_path,
            .error_name = @errorName(err),
        };
    };

    return .{ .path = manifest_path };
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

fn printJsonString(value: []const u8) void {
    std.debug.print("\"", .{});
    for (value) |char| {
        switch (char) {
            '"' => std.debug.print("\\\"", .{}),
            '\\' => std.debug.print("\\\\", .{}),
            '\n' => std.debug.print("\\n", .{}),
            '\r' => std.debug.print("\\r", .{}),
            '\t' => std.debug.print("\\t", .{}),
            else => std.debug.print("{c}", .{char}),
        }
    }
    std.debug.print("\"", .{});
}

test "parseByteCount supports binary suffixes" {
    try std.testing.expectEqual(@as(usize, 64 * 1024), try parseByteCount("64KiB"));
    try std.testing.expectEqual(@as(usize, 2 * 1024 * 1024), try parseByteCount("2MiB"));
    try std.testing.expectEqual(@as(usize, 512), try parseByteCount("512"));
}
