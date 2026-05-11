const std = @import("std");
const onnx = @import("proto/onnx.pb.zig");
const capabilities = @import("capabilities.zig");
const network = @import("network.zig");
const scope = @import("scope.zig");
const target_profile = @import("target.zig");
const workload = @import("workload.zig");
const wasm_compatibility = @import("wasm/compatibility.zig");
const wasm_component = @import("wasm/component.zig");
const wasm_manifest = @import("wasm/manifest.zig");
const wasm_module = @import("wasm/module.zig");

const wasm_page_size: usize = 64 * 1024;

pub const ArtifactKind = enum {
    auto,
    onnx,
    wasm,
    workload,
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

const OnnxAnalysis = struct {
    path: []const u8,
    byte_len: usize,
    decode_error: ?[]const u8 = null,
    model: ?onnx.ModelProto = null,
    report: ?capabilities.Report = null,
    ok: bool = false,

    fn deinit(self: *OnnxAnalysis, allocator: std.mem.Allocator) void {
        if (self.report) |*report| {
            report.deinit(allocator);
        }
        if (self.model) |*model| {
            model.deinit(allocator);
        }
        self.* = undefined;
    }
};

pub fn run(allocator: std.mem.Allocator, options: Options) !bool {
    const kind = try detectKind(options.artifact_path, options.kind);

    return switch (kind) {
        .onnx => try checkOnnxPath(allocator, options.artifact_path, options.output_format),
        .wasm => try checkWasmPath(allocator, options),
        .workload => try checkWorkloadPath(allocator, options),
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

fn checkWorkloadPath(allocator: std.mem.Allocator, options: Options) !bool {
    var loaded = workload.Loaded.load(allocator, options.artifact_path) catch |err| {
        if (options.output_format == .json) {
            printWorkloadLoadFailureJson(options.artifact_path, err);
        } else {
            std.debug.print("zug check workload\n", .{});
            std.debug.print("  path: {s}\n", .{options.artifact_path});
            std.debug.print("  manifest: fail ({s})\n", .{@errorName(err)});
            std.debug.print("  status: fail\n", .{});
        }
        return false;
    };
    defer loaded.deinit(allocator);

    const profile = target_profile.resolve(loaded.manifest.targetName()) catch |err| {
        if (options.output_format == .json) {
            printWorkloadProfileFailureJson(&loaded, err);
        } else {
            printWorkloadHeader(loaded, "unknown");
            std.debug.print("  profile: fail ({s})\n", .{@errorName(err)});
            std.debug.print("  status: fail\n", .{});
        }
        return false;
    };

    const manifest_error = blk: {
        loaded.manifest.validateForTarget(profile) catch |err| break :blk @errorName(err);
        break :blk null;
    };

    var network_report = try network.analyze(allocator, loaded.manifest.network, profile.network);
    defer network_report.deinit(allocator);

    const wasm_bytes = try std.Io.Dir.cwd().readFileAlloc(
        std.Options.debug_io,
        loaded.wasm_path,
        allocator,
        .limited(scope.max_wasm_bytes),
    );
    defer allocator.free(wasm_bytes);

    var wasm_report = try wasm_compatibility.analyzeBytes(allocator, wasm_bytes, compatibilityTarget(profile), .{
        .initial_memory_bytes = options.initial_memory_bytes orelse loaded.manifest.requires.memory_bytes,
        .export_name = options.export_name orelse loaded.manifest.entrypointName(),
    });
    defer wasm_report.deinit(allocator);

    const missing_ml_import = loaded.manifest.requires.wasi_nn and !hasMlImport(wasm_report);

    var model_analysis: ?OnnxAnalysis = null;
    defer if (model_analysis) |*analysis| analysis.deinit(allocator);

    var model_size_error = false;
    if (loaded.model_path) |model_path| {
        model_analysis = try analyzeOnnxPathDetailed(allocator, model_path);
        model_size_error = model_analysis.?.byte_len > profile.max_model_bytes;
    }

    const model_ok = if (model_analysis) |analysis| analysis.ok and !model_size_error else !loaded.manifest.requires.wasi_nn;
    const ok = manifest_error == null and
        !missing_ml_import and
        network_report.supported() and
        wasm_report.supported() and
        model_ok;

    if (options.output_format == .json) {
        printWorkloadCheckJson(
            loaded,
            profile,
            wasm_bytes.len,
            wasm_report,
            manifest_error,
            missing_ml_import,
            network_report,
            model_analysis,
            model_size_error,
            ok,
        );
    } else {
        printWorkloadHeader(loaded, profile.name);
        if (manifest_error) |err| {
            std.debug.print("  manifest: fail ({s})\n", .{err});
        } else {
            std.debug.print("  manifest: pass\n", .{});
        }
        if (missing_ml_import) {
            std.debug.print("  requirement: fail (missing wasi_nn or zug_nn import)\n", .{});
        }
        printNetworkReportText(network_report);

        std.debug.print("  wasm bytes: {d}\n", .{wasm_bytes.len});
        wasm_report.print();

        if (model_analysis) |analysis| {
            printWorkloadModelText(analysis, model_size_error);
        } else {
            std.debug.print("  model: none\n", .{});
        }

        std.debug.print("  status: {s}\n", .{if (ok) "pass" else "fail"});
    }

    return ok;
}

fn checkWasmBytes(allocator: std.mem.Allocator, options: Options, bytes: []const u8) !bool {
    if (wasm_component.isComponent(bytes)) {
        return checkComponentBytes(allocator, options, bytes);
    }

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

fn checkComponentBytes(allocator: std.mem.Allocator, options: Options, bytes: []const u8) !bool {
    var parsed = wasm_component.Component.parse(allocator, bytes) catch |err| {
        if (options.output_format == .json) {
            printComponentCheckJson(options.artifact_path, bytes.len, null, err);
        } else {
            std.debug.print("zug check wasm\n", .{});
            std.debug.print("  file: {s}\n", .{options.artifact_path});
            std.debug.print("  bytes: {d}\n", .{bytes.len});
            std.debug.print("  format: component\n", .{});
            std.debug.print("  parse: fail ({s})\n", .{@errorName(err)});
            std.debug.print("  status: fail\n", .{});
        }
        return false;
    };
    defer parsed.deinit(allocator);

    if (options.output_format == .json) {
        printComponentCheckJson(options.artifact_path, bytes.len, parsed, null);
    } else {
        std.debug.print("zug check wasm\n", .{});
        std.debug.print("  file: {s}\n", .{options.artifact_path});
        std.debug.print("  bytes: {d}\n", .{bytes.len});
        std.debug.print("  format: component\n", .{});
        std.debug.print("  parse: pass\n", .{});
        printComponentSectionCounts(parsed);
        printComponentImports(parsed);
        printComponentExports(parsed);
        std.debug.print("  execution: unsupported (ComponentModelExecutionUnsupported)\n", .{});
        if (options.manifest_path != null) {
            std.debug.print("  manifest: skipped (component execution unsupported)\n", .{});
        }
        std.debug.print("  status: fail\n", .{});
    }

    return false;
}

fn printComponentSectionCounts(parsed: wasm_component.Component) void {
    std.debug.print("  component_sections:\n", .{});
    var any = false;
    for (0..parsed.section_counts.len) |index| {
        const count = parsed.section_counts[index];
        if (count == 0) continue;

        any = true;
        const id: wasm_component.SectionId = @enumFromInt(index);
        std.debug.print("    {s}: {d}\n", .{ wasm_component.sectionName(id), count });
    }
    if (!any) std.debug.print("    none\n", .{});
}

fn printComponentImports(parsed: wasm_component.Component) void {
    std.debug.print("  component_imports: {d}\n", .{parsed.imports.items.len});
    for (parsed.imports.items) |import| {
        std.debug.print("    {s}: {s}", .{
            import.name,
            wasm_component.descName(import.desc),
        });
        if (wasm_component.descTypeIndex(import.desc)) |type_index| {
            std.debug.print(" type={d}", .{type_index});
        }
        if (import.version) |version| {
            std.debug.print(" version={s}", .{version});
        }
        std.debug.print("\n", .{});
    }
}

fn printComponentExports(parsed: wasm_component.Component) void {
    std.debug.print("  component_exports: {d}\n", .{parsed.exports.items.len});
    for (parsed.exports.items) |exported| {
        std.debug.print("    {s}: {s} index={d}", .{
            exported.name,
            wasm_component.sortName(exported.sort_index.sort),
            exported.sort_index.index,
        });
        if (exported.desc) |desc| {
            std.debug.print(" desc={s}", .{wasm_component.descName(desc)});
            if (wasm_component.descTypeIndex(desc)) |type_index| {
                std.debug.print(" type={d}", .{type_index});
            }
        }
        if (exported.version) |version| {
            std.debug.print(" version={s}", .{version});
        }
        std.debug.print("\n", .{});
    }
}

fn printComponentCheckJson(
    artifact_path: []const u8,
    byte_len: usize,
    parsed: ?wasm_component.Component,
    parse_error: ?anyerror,
) void {
    std.debug.print("{{\"kind\":\"wasm\",\"file\":", .{});
    printJsonString(artifact_path);
    std.debug.print(",\"bytes\":{d},\"status\":\"fail\",\"component\":{{\"format\":\"component\",\"supported\":false,\"parse_error\":", .{byte_len});
    printJsonError(parse_error);
    std.debug.print(",\"execution_error\":\"ComponentModelExecutionUnsupported\",\"sections\":{{", .{});
    if (parsed) |component| {
        var first = true;
        for (0..component.section_counts.len) |index| {
            const count = component.section_counts[index];
            if (count == 0) continue;
            if (!first) std.debug.print(",", .{});
            first = false;

            const id: wasm_component.SectionId = @enumFromInt(index);
            printJsonString(wasm_component.sectionName(id));
            std.debug.print(":{d}", .{count});
        }
    }
    std.debug.print("}},\"imports\":[", .{});
    if (parsed) |component| {
        for (component.imports.items, 0..) |import, index| {
            if (index != 0) std.debug.print(",", .{});
            std.debug.print("{{\"name\":", .{});
            printJsonString(import.name);
            std.debug.print(",\"kind\":", .{});
            printJsonString(wasm_component.descName(import.desc));
            std.debug.print(",\"type_index\":", .{});
            printJsonOptionalU32(wasm_component.descTypeIndex(import.desc));
            std.debug.print(",\"version\":", .{});
            printJsonOptionalString(import.version);
            std.debug.print("}}", .{});
        }
    }
    std.debug.print("],\"exports\":[", .{});
    if (parsed) |component| {
        for (component.exports.items, 0..) |exported, index| {
            if (index != 0) std.debug.print(",", .{});
            std.debug.print("{{\"name\":", .{});
            printJsonString(exported.name);
            std.debug.print(",\"kind\":", .{});
            printJsonString(wasm_component.sortName(exported.sort_index.sort));
            std.debug.print(",\"index\":{d},\"version\":", .{exported.sort_index.index});
            printJsonOptionalString(exported.version);
            std.debug.print(",\"desc\":", .{});
            if (exported.desc) |desc| {
                std.debug.print("{{\"kind\":", .{});
                printJsonString(wasm_component.descName(desc));
                std.debug.print(",\"type_index\":", .{});
                printJsonOptionalU32(wasm_component.descTypeIndex(desc));
                std.debug.print("}}", .{});
            } else {
                std.debug.print("null", .{});
            }
            std.debug.print("}}", .{});
        }
    }
    std.debug.print("]}}}}\n", .{});
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

fn printWorkloadHeader(loaded: workload.Loaded, profile_name: []const u8) void {
    std.debug.print("zug check workload\n", .{});
    std.debug.print("  path: {s}\n", .{loaded.root_path});
    std.debug.print("  manifest: {s}\n", .{loaded.manifest_path});
    if (loaded.manifest.name) |name| {
        std.debug.print("  name: {s}\n", .{name});
    }
    std.debug.print("  target: {s}\n", .{profile_name});
    std.debug.print("  entrypoint: {s}\n", .{loaded.manifest.entrypointName()});
    std.debug.print("  wasm: {s}\n", .{loaded.wasm_path});
    if (loaded.model_path) |model_path| {
        std.debug.print("  model: {s}\n", .{model_path});
    }
}

fn printWorkloadModelText(analysis: OnnxAnalysis, model_size_error: bool) void {
    std.debug.print("  model bytes: {d}\n", .{analysis.byte_len});
    if (model_size_error) {
        std.debug.print("  model size: fail (WorkloadModelTooLarge)\n", .{});
    }
    if (analysis.decode_error) |err| {
        std.debug.print("  model decode: fail ({s})\n", .{err});
        return;
    }

    std.debug.print("  model decode: pass\n", .{});
    if (analysis.report) |report| {
        printOnnxReport(report);
    }
}

fn printNetworkReportText(report: network.Report) void {
    std.debug.print("  network:\n", .{});
    if (report.protocol_checks.items.len == 0 and
        report.requested_max_inflight == null and
        report.requested_timeout_ms == null and
        !report.public_egress_requested)
    {
        std.debug.print("    no network requirements\n", .{});
        return;
    }

    for (report.protocol_checks.items) |check_item| {
        std.debug.print("    {s}: {s} {s}\n", .{
            check_item.subject,
            network.protocolName(check_item.protocol),
            if (check_item.supported) "supported" else "unsupported",
        });
    }

    if (report.requested_max_inflight) |requested| {
        std.debug.print("    max_inflight: {d}/{d} {s}\n", .{
            requested,
            report.target_max_inflight,
            if (report.max_inflight_supported) "supported" else "unsupported",
        });
    }
    if (report.requested_timeout_ms) |requested| {
        std.debug.print("    request_timeout_ms: {d}/{d} {s}\n", .{
            requested,
            report.target_max_timeout_ms,
            if (report.timeout_supported) "supported" else "unsupported",
        });
    }
    if (report.public_egress_requested) {
        std.debug.print("    public_egress: {s}\n", .{
            if (report.public_egress_supported) "supported" else "unsupported",
        });
    }
}

fn printWorkloadLoadFailureJson(path: []const u8, err: anyerror) void {
    std.debug.print("{{\"kind\":\"workload\",\"path\":", .{});
    printJsonString(path);
    std.debug.print(",\"manifest\":{{\"status\":\"fail\",\"error\":", .{});
    printJsonString(@errorName(err));
    std.debug.print("}},\"status\":\"fail\"}}\n", .{});
}

fn printWorkloadProfileFailureJson(loaded: *const workload.Loaded, err: anyerror) void {
    std.debug.print("{{\"kind\":\"workload\",\"path\":", .{});
    printJsonString(loaded.root_path);
    std.debug.print(",\"manifest\":", .{});
    printWorkloadManifestJson(loaded.*);
    std.debug.print(",\"target\":{{\"status\":\"fail\",\"error\":", .{});
    printJsonString(@errorName(err));
    std.debug.print("}},\"status\":\"fail\"}}\n", .{});
}

fn printWorkloadCheckJson(
    loaded: workload.Loaded,
    profile: target_profile.Profile,
    wasm_byte_len: usize,
    wasm_report: wasm_compatibility.Report,
    manifest_error: ?[]const u8,
    missing_ml_import: bool,
    network_report: network.Report,
    model_analysis: ?OnnxAnalysis,
    model_size_error: bool,
    ok: bool,
) void {
    std.debug.print("{{\"kind\":\"workload\",\"path\":", .{});
    printJsonString(loaded.root_path);
    std.debug.print(",\"status\":\"{s}\",\"manifest\":", .{if (ok) "pass" else "fail"});
    printWorkloadManifestJson(loaded);
    std.debug.print(",\"target\":", .{});
    printTargetProfileJson(profile);
    std.debug.print(",\"checks\":{{\"manifest_error\":", .{});
    printJsonOptionalString(manifest_error);
    std.debug.print(",\"missing_ml_import\":{}}}", .{missing_ml_import});
    std.debug.print(",\"network\":", .{});
    printNetworkReportJson(network_report);

    std.debug.print(",\"wasm\":{{\"file\":", .{});
    printJsonString(loaded.wasm_path);
    std.debug.print(",\"bytes\":{d},\"compatibility\":", .{wasm_byte_len});
    wasm_report.printJson();
    std.debug.print("}}", .{});

    std.debug.print(",\"model\":", .{});
    if (model_analysis) |analysis| {
        printOnnxAnalysisJson(analysis, model_size_error);
    } else {
        std.debug.print("null", .{});
    }

    std.debug.print("}}\n", .{});
}

fn printWorkloadManifestJson(loaded: workload.Loaded) void {
    std.debug.print("{{\"path\":", .{});
    printJsonString(loaded.manifest_path);
    std.debug.print(",\"name\":", .{});
    printJsonOptionalString(loaded.manifest.name);
    std.debug.print(",\"entrypoint\":", .{});
    printJsonString(loaded.manifest.entrypointName());
    std.debug.print(",\"wasm\":", .{});
    printJsonString(loaded.wasm_path);
    std.debug.print(",\"model\":", .{});
    printJsonOptionalString(loaded.model_path);
    std.debug.print(",\"requires\":{{\"wasi_nn\":{},\"wasi_http\":{},\"memory_bytes\":", .{
        loaded.manifest.requires.wasi_nn,
        loaded.manifest.requires.wasi_http,
    });
    if (loaded.manifest.requires.memory_bytes) |bytes| {
        std.debug.print("{d}", .{bytes});
    } else {
        std.debug.print("null", .{});
    }
    std.debug.print(",\"target\":", .{});
    printJsonOptionalString(loaded.manifest.requires.target_name);
    std.debug.print("}},\"network\":", .{});
    printNetworkRequirementsJson(loaded.manifest.network);
    std.debug.print("}}", .{});
}

fn printTargetProfileJson(profile: target_profile.Profile) void {
    std.debug.print("{{\"name\":", .{});
    printJsonString(profile.name);
    std.debug.print(",\"default_memory_bytes\":{d},\"max_memory_bytes\":{d},\"max_model_bytes\":{d},\"supports_wasi_nn\":{},\"supports_wasi_http\":{},\"network\":", .{
        profile.default_memory_bytes,
        profile.max_memory_bytes,
        profile.max_model_bytes,
        profile.supports_wasi_nn,
        profile.supports_wasi_http,
    });
    printNetworkCapabilitiesJson(profile.network);
    std.debug.print("}}", .{});
}

fn printNetworkRequirementsJson(requirements: network.Requirements) void {
    std.debug.print("{{\"input\":", .{});
    printJsonOptionalProtocol(requirements.input);
    std.debug.print(",\"output\":", .{});
    printJsonOptionalProtocol(requirements.output);
    std.debug.print(",\"ingress\":[", .{});
    for (requirements.ingress.items, 0..) |protocol, index| {
        if (index != 0) std.debug.print(",", .{});
        printJsonString(network.protocolName(protocol));
    }
    std.debug.print("],\"egress\":[", .{});
    for (requirements.egress.items, 0..) |protocol, index| {
        if (index != 0) std.debug.print(",", .{});
        printJsonString(network.protocolName(protocol));
    }
    std.debug.print("],\"max_inflight\":", .{});
    if (requirements.max_inflight) |value| {
        std.debug.print("{d}", .{value});
    } else {
        std.debug.print("null", .{});
    }
    std.debug.print(",\"request_timeout_ms\":", .{});
    if (requirements.request_timeout_ms) |value| {
        std.debug.print("{d}", .{value});
    } else {
        std.debug.print("null", .{});
    }
    std.debug.print(",\"public_egress\":{}}}", .{requirements.public_egress});
}

fn printNetworkCapabilitiesJson(caps: network.Capabilities) void {
    std.debug.print("{{\"protocols\":[", .{});
    for (caps.protocols, 0..) |protocol, index| {
        if (index != 0) std.debug.print(",", .{});
        printJsonString(network.protocolName(protocol));
    }
    std.debug.print("],\"max_inflight\":{d},\"max_request_timeout_ms\":{d},\"public_egress\":{}}}", .{
        caps.max_inflight,
        caps.max_request_timeout_ms,
        caps.public_egress,
    });
}

fn printNetworkReportJson(report: network.Report) void {
    std.debug.print("{{\"supported\":{},\"unsupported_count\":{d},\"protocols\":[", .{
        report.supported(),
        report.unsupported_count,
    });
    for (report.protocol_checks.items, 0..) |check_item, index| {
        if (index != 0) std.debug.print(",", .{});
        std.debug.print("{{\"subject\":", .{});
        printJsonString(check_item.subject);
        std.debug.print(",\"protocol\":", .{});
        printJsonString(network.protocolName(check_item.protocol));
        std.debug.print(",\"supported\":{}}}", .{check_item.supported});
    }
    std.debug.print("],\"max_inflight\":{{\"requested\":", .{});
    if (report.requested_max_inflight) |value| {
        std.debug.print("{d}", .{value});
    } else {
        std.debug.print("null", .{});
    }
    std.debug.print(",\"target\":{d},\"supported\":{}}},\"request_timeout_ms\":{{\"requested\":", .{
        report.target_max_inflight,
        report.max_inflight_supported,
    });
    if (report.requested_timeout_ms) |value| {
        std.debug.print("{d}", .{value});
    } else {
        std.debug.print("null", .{});
    }
    std.debug.print(",\"target\":{d},\"supported\":{}}},\"public_egress\":{{\"requested\":{},\"supported\":{}}}}}", .{
        report.target_max_timeout_ms,
        report.timeout_supported,
        report.public_egress_requested,
        report.public_egress_supported,
    });
}

fn printOnnxAnalysisJson(analysis: OnnxAnalysis, model_size_error: bool) void {
    std.debug.print("{{\"file\":", .{});
    printJsonString(analysis.path);
    std.debug.print(",\"bytes\":{d},\"decode_error\":", .{analysis.byte_len});
    printJsonOptionalString(analysis.decode_error);
    std.debug.print(",\"model_size_error\":{},\"supported\":{}", .{ model_size_error, analysis.ok and !model_size_error });

    if (analysis.report) |report| {
        std.debug.print(",\"operators\":[", .{});
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
        std.debug.print("]", .{});
    }

    std.debug.print("}}", .{});
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

fn analyzeOnnxPathDetailed(allocator: std.mem.Allocator, path: []const u8) !OnnxAnalysis {
    const bytes = try std.Io.Dir.cwd().readFileAlloc(
        std.Options.debug_io,
        path,
        allocator,
        .limited(scope.max_model_bytes),
    );
    defer allocator.free(bytes);

    var analysis = OnnxAnalysis{
        .path = path,
        .byte_len = bytes.len,
    };
    errdefer analysis.deinit(allocator);

    var reader: std.Io.Reader = .fixed(bytes);
    analysis.model = onnx.ModelProto.decode(&reader, allocator) catch |err| {
        analysis.decode_error = @errorName(err);
        return analysis;
    };

    analysis.report = try capabilities.analyze(allocator, &analysis.model.?);
    analysis.ok = onnxReportPassesCheck(analysis.report.?);

    return analysis;
}

fn compatibilityTarget(profile: target_profile.Profile) wasm_compatibility.Target {
    return .{
        .default_memory_bytes = profile.default_memory_bytes,
        .max_memory_bytes = profile.max_memory_bytes,
    };
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
    if (isDirectoryPath(path)) return .workload;
    if (endsWithIgnoreCase(path, ".onnx")) return .onnx;
    if (endsWithIgnoreCase(path, ".wasm")) return .wasm;
    return error.UnknownCheckArtifactKind;
}

fn isDirectoryPath(path: []const u8) bool {
    const stat = std.Io.Dir.cwd().statFile(std.Options.debug_io, path, .{}) catch return false;
    return stat.kind == .directory;
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

fn printJsonError(value: ?anyerror) void {
    if (value) |actual| {
        printJsonString(@errorName(actual));
    } else {
        std.debug.print("null", .{});
    }
}

fn printJsonOptionalString(value: ?[]const u8) void {
    if (value) |actual| {
        printJsonString(actual);
    } else {
        std.debug.print("null", .{});
    }
}

fn printJsonOptionalU32(value: ?u32) void {
    if (value) |actual| {
        std.debug.print("{d}", .{actual});
    } else {
        std.debug.print("null", .{});
    }
}

fn printJsonOptionalProtocol(value: ?network.Protocol) void {
    if (value) |actual| {
        printJsonString(network.protocolName(actual));
    } else {
        std.debug.print("null", .{});
    }
}

test "parseByteCount supports binary suffixes" {
    try std.testing.expectEqual(@as(usize, 64 * 1024), try parseByteCount("64KiB"));
    try std.testing.expectEqual(@as(usize, 2 * 1024 * 1024), try parseByteCount("2MiB"));
    try std.testing.expectEqual(@as(usize, 512), try parseByteCount("512"));
}
