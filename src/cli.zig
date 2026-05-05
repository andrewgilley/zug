const std = @import("std");
const builtin = @import("builtin");
const onnx = @import("proto/onnx.pb.zig");
const tensor = @import("tensor.zig");
const executor = @import("executor.zig");
const capabilities = @import("capabilities.zig");
const benchmark = @import("benchmark.zig");
const check = @import("check.zig");
const scope = @import("scope.zig");
const wasi_nn_abi = @import("wasi_nn_abi.zig");
const wasm_runtime = @import("wasm/runtime.zig");
const wasm_compatibility = @import("wasm/compatibility.zig");
const wasm_imports = @import("wasm/imports.zig");
const wasm_interpreter = @import("wasm/interpreter.zig");
const wasm_manifest = @import("wasm/manifest.zig");

const max_model_bytes = 100 * 1024 * 1024;
const max_wasm_bytes = 100 * 1024 * 1024;
const max_tensor_file_bytes = 100 * 1024 * 1024;
const wasm_page_size: usize = 64 * 1024;
const default_wasm_memory_bytes: usize = 16 * 1024 * 1024;

const CliMode = enum {
    run,
    check,
    inspect,
    scope,
    bench,
    wasm,
};

const BenchFormat = enum {
    text,
    json,
};

const CliInput = struct {
    name: []const u8,
    path: []const u8,
};

const CliArgs = struct {
    mode: CliMode = .run,
    model_path: []const u8,
    inputs: std.ArrayList(CliInput) = .empty,
    output_writes: std.ArrayList(CliInput) = .empty,
    expected_outputs: std.ArrayList(CliInput) = .empty,
    tolerance: f32 = 0.0001,
    wasm_export_name: ?[]const u8 = null,
    wasm_model_path: ?[]const u8 = null,
    wasm_manifest_path: ?[]const u8 = null,
    wasm_model_offset: ?u32 = null,
    wasm_args: std.ArrayList(u32) = .empty,
    bench_iterations: usize = 10,
    bench_warmup: usize = 1,
    bench_format: BenchFormat = .text,
    bench_trace: bool = false,
    check_kind: check.ArtifactKind = .auto,
    check_memory_bytes: ?usize = null,
    check_output_format: check.OutputFormat = .text,

    pub fn deinit(self: *CliArgs, allocator: std.mem.Allocator) void {
        allocator.free(self.model_path);
        if (self.wasm_export_name) |name| {
            allocator.free(name);
        }
        if (self.wasm_model_path) |path| {
            allocator.free(path);
        }
        if (self.wasm_manifest_path) |path| {
            allocator.free(path);
        }

        for (self.inputs.items) |input| {
            allocator.free(input.name);
            allocator.free(input.path);
        }

        for (self.output_writes.items) |output| {
            allocator.free(output.name);
            allocator.free(output.path);
        }

        for (self.expected_outputs.items) |expected| {
            allocator.free(expected.name);
            allocator.free(expected.path);
        }

        self.inputs.deinit(allocator);
        self.output_writes.deinit(allocator);
        self.expected_outputs.deinit(allocator);
        self.wasm_args.deinit(allocator);
    }
};

pub fn main(init: std.process.Init.Minimal) !void {
    var debug_allocator: std.heap.DebugAllocator(.{}) = .init;
    defer _ = debug_allocator.deinit();

    const allocator = debug_allocator.allocator();

    var cli = try parseArgs(init, allocator);
    defer cli.deinit(allocator);

    if (cli.mode == .check) {
        const ok = try check.run(allocator, .{
            .artifact_path = cli.model_path,
            .kind = cli.check_kind,
            .manifest_path = cli.wasm_manifest_path,
            .model_path = cli.wasm_model_path,
            .export_name = cli.wasm_export_name,
            .initial_memory_bytes = cli.check_memory_bytes,
            .output_format = cli.check_output_format,
        });
        if (!ok) std.process.exit(1);
        return;
    }

    if (cli.mode == .scope) {
        scope.print();
        return;
    }

    if (cli.mode == .wasm) {
        try runWasmModule(allocator, &cli);
        return;
    }

    if (cli.mode == .bench) {
        try runOnnxBenchmark(std.heap.page_allocator, &cli);
        return;
    }

    var model = try loadModel(cli.model_path, allocator);
    defer model.deinit(allocator);

    if (cli.mode == .inspect) {
        var report = try capabilities.analyze(allocator, &model);
        defer report.deinit(allocator);

        capabilities.print(report);
        return;
    }

    if (cli.inputs.items.len == 0) {
        printModelSummary(&model);
        return;
    }

    var run = try executor.Executor.init(allocator, &model);
    defer run.deinit();

    try run.loadInputsFromFiles(cli.inputs.items);
    const outputs = try run.execute();

    try writeRequestedOutputs(allocator, outputs, cli.output_writes.items);
    try compareExpectedOutputs(allocator, outputs, cli.expected_outputs.items, cli.tolerance);

    printOutputs(outputs);
}

fn loadModel(model_path: []const u8, allocator: std.mem.Allocator) !onnx.ModelProto {
    const bytes = try std.Io.Dir.cwd().readFileAlloc(
        std.Options.debug_io,
        model_path,
        allocator,
        .limited(max_model_bytes),
    );

    defer allocator.free(bytes);

    var reader: std.Io.Reader = .fixed(bytes);

    return try onnx.ModelProto.decode(&reader, allocator);
}

fn parseArgs(init: std.process.Init.Minimal, allocator: std.mem.Allocator) !CliArgs {
    var args = try init.args.iterateAllocator(allocator);
    defer args.deinit();

    _ = args.skip();

    const first_arg = args.next() orelse {
        printUsage();
        return error.MissingModelPath;
    };

    if (std.mem.eql(u8, first_arg, "inspect")) {
        const model_arg = args.next() orelse {
            printUsage();
            return error.MissingModelPath;
        };

        var cli = CliArgs{
            .mode = .inspect,
            .model_path = try allocator.dupe(u8, model_arg),
        };
        errdefer cli.deinit(allocator);

        if (args.next() != null) {
            printUsage();
            return error.UnknownArgument;
        }

        return cli;
    }

    if (std.mem.eql(u8, first_arg, "check")) {
        const artifact_arg = args.next() orelse {
            printUsage();
            return error.MissingCheckArtifactPath;
        };

        var cli = CliArgs{
            .mode = .check,
            .model_path = try allocator.dupe(u8, artifact_arg),
        };
        errdefer cli.deinit(allocator);

        while (args.next()) |arg| {
            if (std.mem.eql(u8, arg, "--kind")) {
                const value = args.next() orelse {
                    printUsage();
                    return error.MissingCheckKind;
                };

                cli.check_kind = try parseCheckKind(value);
                continue;
            }

            if (std.mem.eql(u8, arg, "--manifest")) {
                const manifest_path = args.next() orelse {
                    printUsage();
                    return error.MissingWasmManifestPath;
                };

                if (cli.wasm_manifest_path) |old_path| {
                    allocator.free(old_path);
                }
                cli.wasm_manifest_path = try allocator.dupe(u8, manifest_path);
                continue;
            }

            if (std.mem.eql(u8, arg, "--model")) {
                const model_path = args.next() orelse {
                    printUsage();
                    return error.MissingWasmModelPath;
                };

                if (cli.wasm_model_path) |old_path| {
                    allocator.free(old_path);
                }
                cli.wasm_model_path = try allocator.dupe(u8, model_path);
                continue;
            }

            if (std.mem.eql(u8, arg, "--export")) {
                const export_name = args.next() orelse {
                    printUsage();
                    return error.MissingWasmExportName;
                };

                if (cli.wasm_export_name) |old_name| {
                    allocator.free(old_name);
                }
                cli.wasm_export_name = try allocator.dupe(u8, export_name);
                continue;
            }

            if (std.mem.eql(u8, arg, "--memory")) {
                const value = args.next() orelse {
                    printUsage();
                    return error.MissingCheckMemory;
                };

                cli.check_memory_bytes = try check.parseByteCount(value);
                continue;
            }

            if (std.mem.eql(u8, arg, "--json")) {
                cli.check_output_format = .json;
                continue;
            }

            printUsage();
            return error.UnknownArgument;
        }

        return cli;
    }

    if (std.mem.eql(u8, first_arg, "scope")) {
        var cli = CliArgs{
            .mode = .scope,
            .model_path = try allocator.dupe(u8, ""),
        };
        errdefer cli.deinit(allocator);

        if (args.next() != null) {
            printUsage();
            return error.UnknownArgument;
        }

        return cli;
    }

    if (std.mem.eql(u8, first_arg, "bench")) {
        const model_arg = args.next() orelse {
            printUsage();
            return error.MissingModelPath;
        };

        var cli = CliArgs{
            .mode = .bench,
            .model_path = try allocator.dupe(u8, model_arg),
        };
        errdefer cli.deinit(allocator);

        while (args.next()) |arg| {
            if (std.mem.eql(u8, arg, "--input")) {
                const spec = args.next() orelse {
                    printUsage();
                    return error.MissingInputSpec;
                };

                try cli.inputs.append(allocator, try parseInputSpec(allocator, spec));
                continue;
            }

            if (std.mem.eql(u8, arg, "--iterations")) {
                const value = args.next() orelse {
                    printUsage();
                    return error.MissingBenchmarkIterations;
                };

                cli.bench_iterations = try std.fmt.parseInt(usize, value, 10);
                if (cli.bench_iterations == 0) return error.InvalidBenchmarkIterations;
                continue;
            }

            if (std.mem.eql(u8, arg, "--warmup")) {
                const value = args.next() orelse {
                    printUsage();
                    return error.MissingBenchmarkWarmup;
                };

                cli.bench_warmup = try std.fmt.parseInt(usize, value, 10);
                continue;
            }

            if (std.mem.eql(u8, arg, "--format")) {
                const value = args.next() orelse {
                    printUsage();
                    return error.MissingBenchmarkFormat;
                };

                cli.bench_format = try parseBenchFormat(value);
                continue;
            }

            if (std.mem.eql(u8, arg, "--trace")) {
                cli.bench_trace = true;
                continue;
            }

            printUsage();
            return error.UnknownArgument;
        }

        return cli;
    }

    if (std.mem.eql(u8, first_arg, "wasm")) {
        const module_arg = args.next() orelse {
            printUsage();
            return error.MissingWasmModulePath;
        };

        var cli = CliArgs{
            .mode = .wasm,
            .model_path = try allocator.dupe(u8, module_arg),
        };
        errdefer cli.deinit(allocator);

        while (args.next()) |arg| {
            if (std.mem.eql(u8, arg, "--export")) {
                const export_name = args.next() orelse {
                    printUsage();
                    return error.MissingWasmExportName;
                };

                if (cli.wasm_export_name) |old_name| {
                    allocator.free(old_name);
                }
                cli.wasm_export_name = try allocator.dupe(u8, export_name);
                continue;
            }

            if (std.mem.eql(u8, arg, "--arg")) {
                const value = args.next() orelse {
                    printUsage();
                    return error.MissingWasmArgument;
                };

                try cli.wasm_args.append(allocator, try parseWasmI32Arg(value));
                continue;
            }

            if (std.mem.eql(u8, arg, "--model")) {
                const model_path = args.next() orelse {
                    printUsage();
                    return error.MissingWasmModelPath;
                };

                if (cli.wasm_model_path) |old_path| {
                    allocator.free(old_path);
                }
                cli.wasm_model_path = try allocator.dupe(u8, model_path);
                continue;
            }

            if (std.mem.eql(u8, arg, "--manifest")) {
                const manifest_path = args.next() orelse {
                    printUsage();
                    return error.MissingWasmManifestPath;
                };

                if (cli.wasm_manifest_path) |old_path| {
                    allocator.free(old_path);
                }
                cli.wasm_manifest_path = try allocator.dupe(u8, manifest_path);
                continue;
            }

            if (std.mem.eql(u8, arg, "--model-offset")) {
                const value = args.next() orelse {
                    printUsage();
                    return error.MissingWasmModelOffset;
                };

                cli.wasm_model_offset = try std.fmt.parseInt(u32, value, 10);
                continue;
            }

            printUsage();
            return error.UnknownArgument;
        }

        return cli;
    }

    var cli = CliArgs{
        .model_path = try allocator.dupe(u8, first_arg),
    };
    errdefer cli.deinit(allocator);

    while (args.next()) |arg| {
        if (std.mem.eql(u8, arg, "--input")) {
            const spec = args.next() orelse {
                printUsage();
                return error.MissingModelPath;
            };

            try cli.inputs.append(allocator, try parseInputSpec(allocator, spec));
            continue;
        }

        if (std.mem.eql(u8, arg, "--output")) {
            const spec = args.next() orelse {
                printUsage();
                return error.MissingOutputSpec;
            };

            try cli.output_writes.append(allocator, try parseInputSpec(allocator, spec));
            continue;
        }

        if (std.mem.eql(u8, arg, "--expect")) {
            const spec = args.next() orelse {
                printUsage();
                return error.MissingExpectedOutputSpec;
            };

            try cli.expected_outputs.append(allocator, try parseInputSpec(allocator, spec));
            continue;
        }

        if (std.mem.eql(u8, arg, "--tolerance")) {
            const value = args.next() orelse {
                printUsage();
                return error.MissingTolerance;
            };

            cli.tolerance = try std.fmt.parseFloat(f32, value);
            if (cli.tolerance < 0) return error.InvalidTolerance;
            continue;
        }

        printUsage();
        return error.UnknownArgument;
    }

    return cli;
}

fn parseWasmI32Arg(value: []const u8) !u32 {
    const parsed = try std.fmt.parseInt(i32, value, 10);
    return @bitCast(parsed);
}

fn parseBenchFormat(value: []const u8) !BenchFormat {
    if (std.mem.eql(u8, value, "text")) return .text;
    if (std.mem.eql(u8, value, "json")) return .json;
    return error.InvalidBenchmarkFormat;
}

fn parseCheckKind(value: []const u8) !check.ArtifactKind {
    if (std.mem.eql(u8, value, "auto")) return .auto;
    if (std.mem.eql(u8, value, "onnx")) return .onnx;
    if (std.mem.eql(u8, value, "wasm")) return .wasm;
    return error.InvalidCheckKind;
}

fn parseInputSpec(allocator: std.mem.Allocator, spec: []const u8) !CliInput {
    const equals_index = std.mem.indexOfScalar(u8, spec, '=') orelse {
        return error.InvalidInputSpec;
    };

    if (equals_index == 0 or equals_index + 1 >= spec.len) {
        return error.InvalidInputSpec;
    }

    const name = try allocator.dupe(u8, spec[0..equals_index]);
    errdefer allocator.free(name);

    const path = try allocator.dupe(u8, spec[equals_index + 1 ..]);
    errdefer allocator.free(path);

    return .{
        .name = name,
        .path = path,
    };
}

fn printUsage() void {
    std.debug.print(
        \\usage:
        \\  zug scope
        \\  zug check <file.onnx|file.wasm> [--kind auto|onnx|wasm] [--manifest manifest] [--model model.onnx] [--export name] [--memory bytes] [--json]
        \\  zug inspect <model.onnx>
        \\  zug bench <model.onnx> --input name=file.f32 [--warmup count] [--iterations count] [--format text|json] [--trace]
        \\  zug <model.onnx> [--input name=file.f32] [--output name=file.raw] [--expect name=file.raw] [--tolerance value]
        \\  zug wasm <module.wasm> [--manifest manifest] [--export name] [--model model.onnx] [--model-offset bytes] [--arg i32]
        \\
        \\onnx:
        \\  scope prints the current runtime capability contract
        \\  check reports compatibility against the current runtime scope
        \\  --output writes a named graph output as raw little-endian tensor bytes
        \\  --expect compares a named graph output against raw little-endian tensor bytes
        \\  --tolerance sets max absolute difference for float32 expectations
        \\  bench uses a non-debug counting allocator and reports timing plus allocation pressure
        \\
        \\wasm:
        \\  --manifest checks runtime requirements before execution
        \\  --model exposes a host-managed model to zug_nn.load_preloaded_graph
        \\  --model-offset also writes that model into guest memory for legacy guests
        \\
    , .{});
}

const BenchmarkPhase = struct {
    ns: i128,
    alloc: benchmark.AllocationDelta,
};

const BenchmarkSample = struct {
    ns: i128,
    alloc: benchmark.AllocationDelta,
};

const OnnxBenchmarkReport = struct {
    model_path: []const u8,
    input_count: usize,
    output_count: usize,
    warmup_iterations: usize,
    measured_iterations: usize,
    load: BenchmarkPhase,
    executor_init: BenchmarkPhase,
    input_bind: BenchmarkPhase,
    warmup: BenchmarkPhase,
    measured_runs: BenchmarkPhase,
    teardown: BenchmarkPhase,
    run_summary: benchmark.TimingSummary,
    active_total_ns: i128,
    total_with_teardown_ns: i128,
    allocations_before_teardown: benchmark.AllocationStats,
    allocations_after_teardown: benchmark.AllocationStats,
    samples: []const BenchmarkSample,
};

fn runOnnxBenchmark(base_allocator: std.mem.Allocator, cli: *const CliArgs) !void {
    if (cli.inputs.items.len == 0) return error.MissingBenchmarkInput;
    if (cli.bench_iterations == 0) return error.InvalidBenchmarkIterations;

    var counter = benchmark.CountingAllocator.init(base_allocator);
    const allocator = counter.allocator();

    const sample_times = try base_allocator.alloc(i128, cli.bench_iterations);
    defer base_allocator.free(sample_times);

    const samples = try base_allocator.alloc(BenchmarkSample, cli.bench_iterations);
    defer base_allocator.free(samples);

    const total_start = benchmark.nowNs();

    const load_stats_start = counter.snapshot();
    const load_start = benchmark.nowNs();
    var model = try loadModel(cli.model_path, allocator);
    const load_end = benchmark.nowNs();
    const load_stats_end = counter.snapshot();
    errdefer model.deinit(allocator);

    const init_stats_start = counter.snapshot();
    const init_start = benchmark.nowNs();
    var run = try executor.Executor.init(allocator, &model);
    const init_end = benchmark.nowNs();
    const init_stats_end = counter.snapshot();
    errdefer run.deinit();

    const input_stats_start = counter.snapshot();
    const input_start = benchmark.nowNs();
    try run.loadInputsFromFiles(cli.inputs.items);
    const input_end = benchmark.nowNs();
    const input_stats_end = counter.snapshot();

    const warmup_stats_start = counter.snapshot();
    const warmup_start = benchmark.nowNs();
    var warmup_index: usize = 0;
    while (warmup_index < cli.bench_warmup) : (warmup_index += 1) {
        _ = try run.execute();
    }
    const warmup_end = benchmark.nowNs();
    const warmup_stats_end = counter.snapshot();

    var output_count: usize = 0;

    const measured_stats_start = counter.snapshot();
    const measured_start = benchmark.nowNs();
    var iteration: usize = 0;
    while (iteration < cli.bench_iterations) : (iteration += 1) {
        const iteration_stats_start = counter.snapshot();
        const run_start = benchmark.nowNs();
        const outputs = try run.execute();
        const run_end = benchmark.nowNs();
        const iteration_stats_end = counter.snapshot();

        output_count = outputs.len;
        const elapsed = run_end - run_start;
        sample_times[iteration] = elapsed;
        samples[iteration] = .{
            .ns = elapsed,
            .alloc = benchmark.allocationDelta(iteration_stats_start, iteration_stats_end),
        };
    }
    const measured_end = benchmark.nowNs();
    const measured_stats_end = counter.snapshot();

    const run_summary = try benchmark.summarizeTimings(base_allocator, sample_times);
    const active_total_end = benchmark.nowNs();
    const allocations_before_teardown = counter.snapshot();

    const teardown_stats_start = counter.snapshot();
    const teardown_start = benchmark.nowNs();
    run.deinit();
    model.deinit(allocator);
    const teardown_end = benchmark.nowNs();
    const teardown_stats_end = counter.snapshot();
    const total_end = benchmark.nowNs();

    const report = OnnxBenchmarkReport{
        .model_path = cli.model_path,
        .input_count = cli.inputs.items.len,
        .output_count = output_count,
        .warmup_iterations = cli.bench_warmup,
        .measured_iterations = cli.bench_iterations,
        .load = .{
            .ns = load_end - load_start,
            .alloc = benchmark.allocationDelta(load_stats_start, load_stats_end),
        },
        .executor_init = .{
            .ns = init_end - init_start,
            .alloc = benchmark.allocationDelta(init_stats_start, init_stats_end),
        },
        .input_bind = .{
            .ns = input_end - input_start,
            .alloc = benchmark.allocationDelta(input_stats_start, input_stats_end),
        },
        .warmup = .{
            .ns = warmup_end - warmup_start,
            .alloc = benchmark.allocationDelta(warmup_stats_start, warmup_stats_end),
        },
        .measured_runs = .{
            .ns = measured_end - measured_start,
            .alloc = benchmark.allocationDelta(measured_stats_start, measured_stats_end),
        },
        .teardown = .{
            .ns = teardown_end - teardown_start,
            .alloc = benchmark.allocationDelta(teardown_stats_start, teardown_stats_end),
        },
        .run_summary = run_summary,
        .active_total_ns = active_total_end - total_start,
        .total_with_teardown_ns = total_end - total_start,
        .allocations_before_teardown = allocations_before_teardown,
        .allocations_after_teardown = counter.snapshot(),
        .samples = samples,
    };

    switch (cli.bench_format) {
        .text => printBenchmarkText(report, cli.bench_trace),
        .json => printBenchmarkJson(report, cli.bench_trace),
    }
}

fn printBenchmarkText(report: OnnxBenchmarkReport, trace: bool) void {
    std.debug.print(
        \\zug benchmark
        \\  model: {s}
        \\  build_mode: {s}
        \\  target: {s}-{s}
        \\  inputs: {d}
        \\  outputs: {d}
        \\  warmup_iterations: {d}
        \\  measured_iterations: {d}
        \\
    , .{
        report.model_path,
        @tagName(builtin.mode),
        @tagName(builtin.target.cpu.arch),
        @tagName(builtin.target.os.tag),
        report.input_count,
        report.output_count,
        report.warmup_iterations,
        report.measured_iterations,
    });

    std.debug.print("phases:\n", .{});
    printBenchmarkPhaseText("model_load", report.load);
    printBenchmarkPhaseText("executor_init", report.executor_init);
    printBenchmarkPhaseText("input_bind", report.input_bind);
    printBenchmarkPhaseText("warmup", report.warmup);
    printBenchmarkPhaseText("measured_runs", report.measured_runs);
    printBenchmarkPhaseText("teardown", report.teardown);

    std.debug.print(
        \\runs:
        \\  avg_ms: {d:.3}
        \\  min_ms: {d:.3}
        \\  p50_ms: {d:.3}
        \\  p95_ms: {d:.3}
        \\  max_ms: {d:.3}
        \\  measured_run_total_ms: {d:.3}
        \\  active_total_ms: {d:.3}
        \\  total_with_teardown_ms: {d:.3}
        \\
    , .{
        benchmark.nsToMs(report.run_summary.avg_ns),
        benchmark.nsToMs(report.run_summary.min_ns),
        benchmark.nsToMs(report.run_summary.p50_ns),
        benchmark.nsToMs(report.run_summary.p95_ns),
        benchmark.nsToMs(report.run_summary.max_ns),
        benchmark.nsToMs(report.run_summary.total_ns),
        benchmark.nsToMs(report.active_total_ns),
        benchmark.nsToMs(report.total_with_teardown_ns),
    });

    std.debug.print(
        \\allocations:
        \\  total_alloc_calls: {d}
        \\  total_free_calls: {d}
        \\  total_resize_calls: {d}
        \\  total_remap_calls: {d}
        \\  total_allocated_bytes: {d}
        \\  total_freed_bytes: {d}
        \\  peak_live_bytes: {d}
        \\  live_before_teardown_bytes: {d}
        \\  live_after_teardown_bytes: {d}
        \\
    , .{
        report.allocations_after_teardown.alloc_calls,
        report.allocations_after_teardown.free_calls,
        report.allocations_after_teardown.resize_calls,
        report.allocations_after_teardown.remap_calls,
        report.allocations_after_teardown.allocated_bytes,
        report.allocations_after_teardown.freed_bytes,
        report.allocations_after_teardown.peak_bytes,
        report.allocations_before_teardown.current_bytes,
        report.allocations_after_teardown.current_bytes,
    });

    if (trace) {
        std.debug.print("samples:\n", .{});
        for (report.samples, 0..) |sample, index| {
            std.debug.print(
                "  {d}: {d:.3} ms, alloc_bytes={d}, alloc_calls={d}, free_calls={d}, net_bytes={d}\n",
                .{
                    index,
                    benchmark.nsToMs(sample.ns),
                    sample.alloc.allocated_bytes,
                    sample.alloc.alloc_calls,
                    sample.alloc.free_calls,
                    sample.alloc.net_bytes,
                },
            );
        }
    }
}

fn printBenchmarkPhaseText(name: []const u8, phase: BenchmarkPhase) void {
    std.debug.print(
        "  {s}: {d:.3} ms, alloc_bytes={d}, alloc_calls={d}, free_calls={d}, net_bytes={d}\n",
        .{
            name,
            benchmark.nsToMs(phase.ns),
            phase.alloc.allocated_bytes,
            phase.alloc.alloc_calls,
            phase.alloc.free_calls,
            phase.alloc.net_bytes,
        },
    );
}

fn printBenchmarkJson(report: OnnxBenchmarkReport, trace: bool) void {
    std.debug.print("{{", .{});
    std.debug.print("\"model\":", .{});
    printJsonString(report.model_path);
    std.debug.print(",\"build_mode\":", .{});
    printJsonString(@tagName(builtin.mode));
    std.debug.print(",\"target\":{{\"arch\":", .{});
    printJsonString(@tagName(builtin.target.cpu.arch));
    std.debug.print(",\"os\":", .{});
    printJsonString(@tagName(builtin.target.os.tag));
    std.debug.print("}}", .{});
    std.debug.print(",\"inputs\":{d}", .{report.input_count});
    std.debug.print(",\"outputs\":{d}", .{report.output_count});
    std.debug.print(",\"warmup_iterations\":{d}", .{report.warmup_iterations});
    std.debug.print(",\"measured_iterations\":{d}", .{report.measured_iterations});

    std.debug.print(",\"phases\":{{", .{});
    printBenchmarkPhaseJson("model_load", report.load, true);
    printBenchmarkPhaseJson("executor_init", report.executor_init, true);
    printBenchmarkPhaseJson("input_bind", report.input_bind, true);
    printBenchmarkPhaseJson("warmup", report.warmup, true);
    printBenchmarkPhaseJson("measured_runs", report.measured_runs, true);
    printBenchmarkPhaseJson("teardown", report.teardown, false);
    std.debug.print("}}", .{});

    std.debug.print(
        ",\"runs\":{{\"avg_ms\":{d:.6},\"min_ms\":{d:.6},\"p50_ms\":{d:.6},\"p95_ms\":{d:.6},\"max_ms\":{d:.6},\"measured_run_total_ms\":{d:.6},\"active_total_ms\":{d:.6},\"total_with_teardown_ms\":{d:.6}}}",
        .{
            benchmark.nsToMs(report.run_summary.avg_ns),
            benchmark.nsToMs(report.run_summary.min_ns),
            benchmark.nsToMs(report.run_summary.p50_ns),
            benchmark.nsToMs(report.run_summary.p95_ns),
            benchmark.nsToMs(report.run_summary.max_ns),
            benchmark.nsToMs(report.run_summary.total_ns),
            benchmark.nsToMs(report.active_total_ns),
            benchmark.nsToMs(report.total_with_teardown_ns),
        },
    );

    std.debug.print(",\"allocations\":", .{});
    printAllocationStatsJson(report.allocations_after_teardown, report.allocations_before_teardown.current_bytes);

    if (trace) {
        std.debug.print(",\"samples\":[", .{});
        for (report.samples, 0..) |sample, index| {
            if (index != 0) std.debug.print(",", .{});
            std.debug.print(
                "{{\"index\":{d},\"ms\":{d:.6},\"alloc_bytes\":{d},\"alloc_calls\":{d},\"free_calls\":{d},\"net_bytes\":{d}}}",
                .{
                    index,
                    benchmark.nsToMs(sample.ns),
                    sample.alloc.allocated_bytes,
                    sample.alloc.alloc_calls,
                    sample.alloc.free_calls,
                    sample.alloc.net_bytes,
                },
            );
        }
        std.debug.print("]", .{});
    }

    std.debug.print("}}\n", .{});
}

fn printBenchmarkPhaseJson(name: []const u8, phase: BenchmarkPhase, trailing_comma: bool) void {
    printJsonString(name);
    std.debug.print(
        ":{{\"ms\":{d:.6},\"alloc_bytes\":{d},\"alloc_calls\":{d},\"free_calls\":{d},\"resize_calls\":{d},\"remap_calls\":{d},\"net_bytes\":{d}}}",
        .{
            benchmark.nsToMs(phase.ns),
            phase.alloc.allocated_bytes,
            phase.alloc.alloc_calls,
            phase.alloc.free_calls,
            phase.alloc.resize_calls,
            phase.alloc.remap_calls,
            phase.alloc.net_bytes,
        },
    );
    if (trailing_comma) std.debug.print(",", .{});
}

fn printAllocationStatsJson(stats: benchmark.AllocationStats, live_before_teardown_bytes: usize) void {
    std.debug.print(
        "{{\"total_alloc_calls\":{d},\"total_free_calls\":{d},\"total_resize_calls\":{d},\"total_remap_calls\":{d},\"total_allocated_bytes\":{d},\"total_freed_bytes\":{d},\"peak_live_bytes\":{d},\"live_before_teardown_bytes\":{d},\"live_after_teardown_bytes\":{d}}}",
        .{
            stats.alloc_calls,
            stats.free_calls,
            stats.resize_calls,
            stats.remap_calls,
            stats.allocated_bytes,
            stats.freed_bytes,
            stats.peak_bytes,
            live_before_teardown_bytes,
            stats.current_bytes,
        },
    );
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

fn runWasmModule(allocator: std.mem.Allocator, cli: *const CliArgs) !void {
    const bytes = try std.Io.Dir.cwd().readFileAlloc(
        std.Options.debug_io,
        cli.model_path,
        allocator,
        .limited(max_wasm_bytes),
    );
    defer allocator.free(bytes);

    const model_bytes = if (cli.wasm_model_path) |model_path|
        try std.Io.Dir.cwd().readFileAlloc(
            std.Options.debug_io,
            model_path,
            allocator,
            .limited(max_model_bytes),
        )
    else
        null;
    defer if (model_bytes) |loaded| allocator.free(loaded);

    const manifest_bytes = if (cli.wasm_manifest_path) |manifest_path|
        try std.Io.Dir.cwd().readFileAlloc(
            std.Options.debug_io,
            manifest_path,
            allocator,
            .limited(wasm_manifest.max_manifest_bytes),
        )
    else
        null;
    defer if (manifest_bytes) |loaded| allocator.free(loaded);

    var manifest: ?wasm_manifest.Manifest = null;
    defer if (manifest) |*loaded| loaded.deinit(allocator);

    if (manifest_bytes) |loaded| {
        manifest = try wasm_manifest.Manifest.parse(allocator, loaded);
    }

    if (cli.wasm_model_offset != null and model_bytes == null) {
        return error.ModelOffsetRequiresModel;
    }

    const runtime = wasm_runtime.Runtime.init(allocator);

    var parsed = try runtime.parseModule(bytes);
    defer parsed.deinit(allocator);

    const export_name = cli.wasm_export_name orelse if (manifest) |loaded|
        loaded.export_name orelse "run"
    else
        "run";

    const initial_memory_bytes = try wasmInitialMemoryBytes(cli, model_bytes, manifest);

    var compatibility_report = try wasm_compatibility.analyzeModule(allocator, bytes, &parsed, .{}, .{
        .initial_memory_bytes = initial_memory_bytes,
        .export_name = export_name,
    });
    defer compatibility_report.deinit(allocator);

    if (!compatibility_report.supported()) {
        std.debug.print("zug wasm compatibility: fail\n", .{});
        compatibility_report.print();
        return error.WasmCompatibilityCheckFailed;
    }

    if (manifest) |loaded| {
        try loaded.validate(&parsed, .{
            .max_model_bytes = max_model_bytes,
        }, .{
            .export_name = export_name,
            .has_preloaded_model = model_bytes != null,
            .model_len = if (model_bytes) |model| model.len else 0,
            .initial_memory_bytes = initial_memory_bytes,
        });
    }

    var wasm_instance = try runtime.instantiate(&parsed, initial_memory_bytes);
    defer wasm_instance.deinit();

    if (model_bytes) |loaded| {
        if (cli.wasm_model_offset) |model_offset| {
            try wasm_instance.memory.write(model_offset, loaded);
        }
    }

    var host = wasi_nn_abi.Host.init(allocator);
    defer host.deinit();

    var surface = wasi_nn_abi.Surface.init(allocator, &host, &wasm_instance.memory);
    surface.setPreloadedModel(model_bytes);

    var resolver = wasm_imports.Resolver.init(&surface);
    defer resolver.deinit();

    try wasm_instance.bindImports(&resolver);

    var interpreter = wasm_interpreter.Interpreter.init(&wasm_instance);
    try interpreter.runStart();

    var args: std.ArrayList(wasm_interpreter.Value) = .empty;
    defer args.deinit(allocator);

    if (model_bytes) |loaded| {
        if (cli.wasm_model_offset != null) {
            const model_len = std.math.cast(u32, loaded.len) orelse return error.ModelTooLarge;
            try args.append(allocator, .{ .i32 = model_len });
        }
    }

    for (cli.wasm_args.items) |arg| {
        try args.append(allocator, .{ .i32 = arg });
    }

    const result = try interpreter.callExport(export_name, args.items);

    if (resolver.stdout.items.len != 0) {
        std.debug.print("{s}", .{resolver.stdout.items});
    }
    if (resolver.stderr.items.len != 0) {
        std.debug.print("{s}", .{resolver.stderr.items});
    }
    if (resolver.exit_code) |exit_code| {
        std.debug.print("wasm exit: {d}\n", .{exit_code});
    }

    printWasmResult(result);
}

fn wasmInitialMemoryBytes(
    cli: *const CliArgs,
    model_bytes: ?[]const u8,
    manifest: ?wasm_manifest.Manifest,
) !usize {
    var needed = default_wasm_memory_bytes;

    if (model_bytes) |loaded| {
        if (cli.wasm_model_offset) |model_offset_u32| {
            const model_offset = std.math.cast(usize, model_offset_u32) orelse return error.ModelTooLarge;
            const model_end = try std.math.add(usize, model_offset, loaded.len);
            needed = @max(needed, model_end);
        }
    }

    if (manifest) |loaded| {
        if (loaded.min_memory_bytes) |min_memory_bytes| {
            needed = @max(needed, min_memory_bytes);
        }
    }

    const aligned = try alignToWasmPage(needed);

    if (manifest) |loaded| {
        if (loaded.max_memory_bytes) |max_memory_bytes| {
            if (aligned > max_memory_bytes) return error.ManifestMemoryTooLarge;
        }
    }

    return aligned;
}

fn alignToWasmPage(value: usize) !usize {
    const plus_page = try std.math.add(usize, value, wasm_page_size - 1);
    return plus_page - (plus_page % wasm_page_size);
}

fn printWasmResult(result: ?wasm_interpreter.Value) void {
    std.debug.print("wasm result: ", .{});

    if (result) |value| {
        switch (value) {
            .i32 => |actual| std.debug.print("i32 {d}", .{@as(i32, @bitCast(actual))}),
            .i64 => |actual| std.debug.print("i64 {d}", .{@as(i64, @bitCast(actual))}),
            .f32 => |actual| std.debug.print("f32 {d}", .{actual}),
            .f64 => |actual| std.debug.print("f64 {d}", .{actual}),
            .v128 => |actual| {
                std.debug.print("v128", .{});
                for (actual) |byte| {
                    std.debug.print(" {x:0>2}", .{byte});
                }
            },
        }
    } else {
        std.debug.print("<none>", .{});
    }

    std.debug.print("\n", .{});
}

fn printModelSummary(model: *const onnx.ModelProto) void {
    std.debug.print("ONNX model decoded\n", .{});

    if (model.ir_version) |ir_version| {
        std.debug.print("ir_version: {d}\n", .{ir_version});
    }

    if (model.producer_name) |producer_name| {
        std.debug.print("producer_name: {s}\n", .{producer_name});
    }

    if (model.producer_version) |producer_version| {
        std.debug.print("producer_version: {s}\n", .{producer_version});
    }

    std.debug.print("opsets: {d}\n", .{model.opset_import.items.len});

    if (model.graph) |*graph| {
        std.debug.print("graph: {s}\n", .{graph.name orelse "<unnamed>"});
        std.debug.print("inputs: {d}\n", .{graph.input.items.len});
        std.debug.print("outputs: {d}\n", .{graph.output.items.len});
        std.debug.print("nodes: {d}\n", .{graph.node.items.len});
        std.debug.print("initializers: {d}\n", .{graph.initializer.items.len});
    } else {
        std.debug.print("graph: <missing>\n", .{});
    }
}

fn printOutputs(outputs: []const executor.Output) void {
    for (outputs) |output| {
        printTensor(output.name, output.value);
    }
}

fn writeRequestedOutputs(
    allocator: std.mem.Allocator,
    outputs: []const executor.Output,
    specs: []const CliInput,
) !void {
    for (specs) |spec| {
        const output = findOutput(outputs, spec.name) orelse return error.UnknownOutputName;
        const bytes = try tensorToRawBytes(allocator, output.value);
        defer allocator.free(bytes);

        try std.Io.Dir.cwd().writeFile(std.Options.debug_io, .{
            .sub_path = spec.path,
            .data = bytes,
        });

        std.debug.print("wrote output {s} -> {s} ({d} bytes)\n", .{ spec.name, spec.path, bytes.len });
    }
}

fn compareExpectedOutputs(
    allocator: std.mem.Allocator,
    outputs: []const executor.Output,
    specs: []const CliInput,
    tolerance: f32,
) !void {
    for (specs) |spec| {
        const output = findOutput(outputs, spec.name) orelse return error.UnknownOutputName;
        const actual_bytes = try tensorToRawBytes(allocator, output.value);
        defer allocator.free(actual_bytes);

        const expected_bytes = try std.Io.Dir.cwd().readFileAlloc(
            std.Options.debug_io,
            spec.path,
            allocator,
            .limited(max_tensor_file_bytes),
        );
        defer allocator.free(expected_bytes);

        if (expected_bytes.len != actual_bytes.len) return error.ExpectedOutputByteLengthMismatch;

        switch (output.value.dtype) {
            .float32 => try compareFloat32Output(spec.name, actual_bytes, expected_bytes, tolerance),
            else => try compareExactOutput(spec.name, actual_bytes, expected_bytes),
        }
    }
}

fn findOutput(outputs: []const executor.Output, name: []const u8) ?executor.Output {
    for (outputs) |output| {
        if (std.mem.eql(u8, output.name, name)) return output;
    }

    return null;
}

fn compareFloat32Output(
    name: []const u8,
    actual_bytes: []const u8,
    expected_bytes: []const u8,
    tolerance: f32,
) !void {
    if (actual_bytes.len % @sizeOf(f32) != 0) return error.InvalidTensorByteLength;

    var max_abs_diff: f32 = 0;
    var mismatches: usize = 0;

    var offset: usize = 0;
    while (offset < actual_bytes.len) : (offset += @sizeOf(f32)) {
        const actual: f32 = @bitCast(readU32Little(actual_bytes[offset..][0..4]));
        const expected: f32 = @bitCast(readU32Little(expected_bytes[offset..][0..4]));
        const diff = @abs(actual - expected);

        if (diff > max_abs_diff) max_abs_diff = diff;
        if (diff > tolerance) mismatches += 1;
    }

    std.debug.print(
        "checked output {s}: max_abs_diff={d}, mismatches={d}, tolerance={d}\n",
        .{ name, max_abs_diff, mismatches, tolerance },
    );

    if (mismatches != 0) return error.ExpectedOutputMismatch;
}

fn compareExactOutput(name: []const u8, actual_bytes: []const u8, expected_bytes: []const u8) !void {
    if (std.mem.eql(u8, actual_bytes, expected_bytes)) {
        std.debug.print("checked output {s}: exact match\n", .{name});
        return;
    }

    return error.ExpectedOutputMismatch;
}

fn tensorToRawBytes(allocator: std.mem.Allocator, value: *const tensor.Tensor) ![]u8 {
    return switch (value.data) {
        .float32 => |values| encodeFloat32Bytes(allocator, values),
        .int64 => |values| encodeInt64Bytes(allocator, values),
        .int32 => |values| encodeInt32Bytes(allocator, values),
        .uint8 => |values| allocator.dupe(u8, values),
        .bool => |values| encodeBoolBytes(allocator, values),
    };
}

fn encodeFloat32Bytes(allocator: std.mem.Allocator, values: []const f32) ![]u8 {
    const bytes = try allocator.alloc(u8, values.len * @sizeOf(f32));
    errdefer allocator.free(bytes);

    for (values, 0..) |value, index| {
        writeU32Little(bytes[index * @sizeOf(f32) ..][0..4], @bitCast(value));
    }

    return bytes;
}

fn encodeInt64Bytes(allocator: std.mem.Allocator, values: []const i64) ![]u8 {
    const bytes = try allocator.alloc(u8, values.len * @sizeOf(i64));
    errdefer allocator.free(bytes);

    for (values, 0..) |value, index| {
        writeU64Little(bytes[index * @sizeOf(i64) ..][0..8], @bitCast(value));
    }

    return bytes;
}

fn encodeInt32Bytes(allocator: std.mem.Allocator, values: []const i32) ![]u8 {
    const bytes = try allocator.alloc(u8, values.len * @sizeOf(i32));
    errdefer allocator.free(bytes);

    for (values, 0..) |value, index| {
        writeU32Little(bytes[index * @sizeOf(i32) ..][0..4], @bitCast(value));
    }

    return bytes;
}

fn encodeBoolBytes(allocator: std.mem.Allocator, values: []const bool) ![]u8 {
    const bytes = try allocator.alloc(u8, values.len);
    errdefer allocator.free(bytes);

    for (values, 0..) |value, index| {
        bytes[index] = if (value) 1 else 0;
    }

    return bytes;
}

fn readU32Little(bytes: *const [4]u8) u32 {
    return @as(u32, bytes[0]) |
        (@as(u32, bytes[1]) << 8) |
        (@as(u32, bytes[2]) << 16) |
        (@as(u32, bytes[3]) << 24);
}

fn writeU32Little(bytes: *[4]u8, value: u32) void {
    bytes[0] = @truncate(value);
    bytes[1] = @truncate(value >> 8);
    bytes[2] = @truncate(value >> 16);
    bytes[3] = @truncate(value >> 24);
}

fn writeU64Little(bytes: *[8]u8, value: u64) void {
    bytes[0] = @truncate(value);
    bytes[1] = @truncate(value >> 8);
    bytes[2] = @truncate(value >> 16);
    bytes[3] = @truncate(value >> 24);
    bytes[4] = @truncate(value >> 32);
    bytes[5] = @truncate(value >> 40);
    bytes[6] = @truncate(value >> 48);
    bytes[7] = @truncate(value >> 56);
}

fn printTensor(name: []const u8, value: *const tensor.Tensor) void {
    std.debug.print("{s} ", .{name});

    printDType(value.dtype);
    printShape(value.shape);

    std.debug.print(" = ", .{});

    printValues(value);

    std.debug.print("\n", .{});
}

fn printDType(dtype: tensor.DType) void {
    switch (dtype) {
        .float32 => std.debug.print("float32", .{}),
        .int64 => std.debug.print("int64", .{}),
        .int32 => std.debug.print("int32", .{}),
        .uint8 => std.debug.print("uint8", .{}),
        .bool => std.debug.print("bool", .{}),
    }
}

fn printShape(shape: []const usize) void {
    std.debug.print("[", .{});

    for (shape, 0..) |dim, index| {
        if (index != 0) std.debug.print(",", .{});
        std.debug.print("{d}", .{dim});
    }

    std.debug.print("]", .{});
}

fn printValues(value: *const tensor.Tensor) void {
    switch (value.data) {
        .float32 => |values| printTypedValues(f32, values),
        .int64 => |values| printTypedValues(i64, values),
        .int32 => |values| printTypedValues(i32, values),
        .uint8 => |values| printTypedValues(u8, values),
        .bool => |values| printTypedValues(bool, values),
    }
}

fn printTypedValues(comptime T: type, values: []const T) void {
    const max_values = 64;
    const shown = @min(values.len, max_values);

    std.debug.print("[", .{});

    for (values[0..shown], 0..) |value, index| {
        if (index != 0) std.debug.print(", ", .{});
        if (T == bool) {
            std.debug.print("{any}", .{value});
        } else {
            std.debug.print("{d}", .{value});
        }
    }

    if (values.len > shown) {
        std.debug.print(", ... {d} total", .{values.len});
    }

    std.debug.print("]", .{});
}
