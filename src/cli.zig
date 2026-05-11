const std = @import("std");
const builtin = @import("builtin");
const onnx = @import("proto/onnx.pb.zig");
const tensor = @import("tensor.zig");
const executor = @import("executor.zig");
const capabilities = @import("capabilities.zig");
const benchmark = @import("benchmark.zig");
const check = @import("check.zig");
const agent = @import("agent.zig");
const accelerator = @import("accelerator.zig");
const gpu = @import("gpu.zig");
const scope = @import("scope.zig");
const wasi_nn_abi = @import("wasi_nn_abi.zig");
const network = @import("network.zig");
const target_profile = @import("target.zig");
const workload = @import("workload.zig");
const workload_runner = @import("workload_runner.zig");
const wit_generator = @import("wit_generator.zig");
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
const max_upload_response_bytes: usize = 1024 * 1024;
const net = std.Io.net;

const CliMode = enum {
    run,
    workload,
    check,
    inspect,
    scope,
    bench,
    wasm,
    agent,
    upload,
    wit,
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
    agent_node_id: ?[]const u8 = null,
    agent_profile_name: ?[]const u8 = null,
    agent_listen: ?[]const u8 = null,
    agent_once: bool = false,
    upload_agent_url: ?[]const u8 = null,
    upload_deploy: bool = false,
    upload_chunk_size: usize = 64 * 1024,
    run_profile_name: ?[]const u8 = null,
    wasm_stdin: ?[]const u8 = null,
    wit_output_path: ?[]const u8 = null,
    mock_gpu: bool = false,

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
        if (self.agent_node_id) |node_id| {
            allocator.free(node_id);
        }
        if (self.agent_profile_name) |profile_name| {
            allocator.free(profile_name);
        }
        if (self.agent_listen) |listen| {
            allocator.free(listen);
        }
        if (self.upload_agent_url) |url| {
            allocator.free(url);
        }
        if (self.run_profile_name) |profile_name| {
            allocator.free(profile_name);
        }
        if (self.wasm_stdin) |stdin| {
            allocator.free(stdin);
        }
        if (self.wit_output_path) |path| {
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

    if (cli.mode == .agent) {
        try agent.serve(allocator, .{
            .node_id = cli.agent_node_id orelse "local-node",
            .profile_name = cli.agent_profile_name orelse "vision-f32-basic",
            .listen = cli.agent_listen orelse "127.0.0.1:7070",
            .once = cli.agent_once,
            .gpu = if (cli.mock_gpu) gpu.mockEdgeCapabilities() else .{},
            .accelerators = if (cli.mock_gpu) accelerator.mockEdgeCapabilities() else accelerator.defaultCapabilities(),
        });
        return;
    }

    if (cli.mode == .upload) {
        try runUpload(allocator, &cli);
        return;
    }

    if (cli.mode == .wit) {
        try wit_generator.run(allocator, .{
            .descriptor = cli.model_path,
            .out_path = cli.wit_output_path,
        });
        return;
    }

    if (cli.mode == .wasm) {
        try runWasmModule(allocator, &cli);
        return;
    }

    if (cli.mode == .workload) {
        try runWorkload(allocator, &cli);
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

    if (std.mem.eql(u8, first_arg, "run")) {
        const workload_arg = args.next() orelse {
            printUsage();
            return error.MissingWorkloadPath;
        };

        var cli = CliArgs{
            .mode = .workload,
            .model_path = try allocator.dupe(u8, workload_arg),
        };
        errdefer cli.deinit(allocator);

        while (args.next()) |arg| {
            if (std.mem.eql(u8, arg, "--arg")) {
                const value = args.next() orelse {
                    printUsage();
                    return error.MissingWasmArgument;
                };

                try cli.wasm_args.append(allocator, try parseWasmI32Arg(value));
                continue;
            }

            if (std.mem.eql(u8, arg, "--stdin")) {
                const value = args.next() orelse {
                    printUsage();
                    return error.MissingWasmStdin;
                };

                if (cli.wasm_stdin) |old_stdin| {
                    allocator.free(old_stdin);
                }
                cli.wasm_stdin = try allocator.dupe(u8, value);
                continue;
            }

            if (std.mem.eql(u8, arg, "--profile")) {
                const profile_name = args.next() orelse {
                    printUsage();
                    return error.MissingRunProfile;
                };

                if (cli.run_profile_name) |old_profile_name| {
                    allocator.free(old_profile_name);
                }
                cli.run_profile_name = try allocator.dupe(u8, profile_name);
                continue;
            }

            if (std.mem.eql(u8, arg, "--json")) {
                cli.check_output_format = .json;
                continue;
            }

            if (std.mem.eql(u8, arg, "--mock-gpu")) {
                cli.mock_gpu = true;
                continue;
            }

            printUsage();
            return error.UnknownArgument;
        }

        return cli;
    }

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

    if (std.mem.eql(u8, first_arg, "agent")) {
        var cli = CliArgs{
            .mode = .agent,
            .model_path = try allocator.dupe(u8, ""),
        };
        errdefer cli.deinit(allocator);

        while (args.next()) |arg| {
            if (std.mem.eql(u8, arg, "--node") or std.mem.eql(u8, arg, "--node-id")) {
                const node_id = args.next() orelse {
                    printUsage();
                    return error.MissingAgentNodeId;
                };

                if (cli.agent_node_id) |old_node_id| {
                    allocator.free(old_node_id);
                }
                cli.agent_node_id = try allocator.dupe(u8, node_id);
                continue;
            }

            if (std.mem.eql(u8, arg, "--profile")) {
                const profile_name = args.next() orelse {
                    printUsage();
                    return error.MissingAgentProfile;
                };

                if (cli.agent_profile_name) |old_profile_name| {
                    allocator.free(old_profile_name);
                }
                cli.agent_profile_name = try allocator.dupe(u8, profile_name);
                continue;
            }

            if (std.mem.eql(u8, arg, "--listen")) {
                const listen = args.next() orelse {
                    printUsage();
                    return error.MissingAgentListenAddress;
                };

                if (cli.agent_listen) |old_listen| {
                    allocator.free(old_listen);
                }
                cli.agent_listen = try allocator.dupe(u8, listen);
                continue;
            }

            if (std.mem.eql(u8, arg, "--once")) {
                cli.agent_once = true;
                continue;
            }

            if (std.mem.eql(u8, arg, "--mock-gpu")) {
                cli.mock_gpu = true;
                continue;
            }

            printUsage();
            return error.UnknownArgument;
        }

        return cli;
    }

    if (std.mem.eql(u8, first_arg, "upload")) {
        const workload_arg = args.next() orelse {
            printUsage();
            return error.MissingWorkloadPath;
        };

        var cli = CliArgs{
            .mode = .upload,
            .model_path = try allocator.dupe(u8, workload_arg),
        };
        errdefer cli.deinit(allocator);

        while (args.next()) |arg| {
            if (std.mem.eql(u8, arg, "--agent")) {
                const url = args.next() orelse {
                    printUsage();
                    return error.MissingUploadAgent;
                };

                if (cli.upload_agent_url) |old_url| {
                    allocator.free(old_url);
                }
                cli.upload_agent_url = try allocator.dupe(u8, url);
                continue;
            }

            if (std.mem.eql(u8, arg, "--deploy")) {
                cli.upload_deploy = true;
                continue;
            }

            if (std.mem.eql(u8, arg, "--chunk-size")) {
                const value = args.next() orelse {
                    printUsage();
                    return error.MissingUploadChunkSize;
                };

                cli.upload_chunk_size = try std.fmt.parseInt(usize, value, 10);
                if (cli.upload_chunk_size == 0) return error.InvalidUploadChunkSize;
                continue;
            }

            printUsage();
            return error.UnknownArgument;
        }

        if (cli.upload_agent_url == null) return error.MissingUploadAgent;
        return cli;
    }

    if (std.mem.eql(u8, first_arg, "wit")) {
        const descriptor_arg = args.next() orelse {
            printUsage();
            return error.MissingWitDescriptor;
        };

        var cli = CliArgs{
            .mode = .wit,
            .model_path = try allocator.dupe(u8, descriptor_arg),
        };
        errdefer cli.deinit(allocator);

        while (args.next()) |arg| {
            if (std.mem.eql(u8, arg, "--out")) {
                const out_path = args.next() orelse {
                    printUsage();
                    return error.MissingWitOutputPath;
                };

                if (cli.wit_output_path) |old_path| {
                    allocator.free(old_path);
                }
                cli.wit_output_path = try allocator.dupe(u8, out_path);
                continue;
            }

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

            if (std.mem.eql(u8, arg, "--mock-gpu")) {
                cli.mock_gpu = true;
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
    if (std.mem.eql(u8, value, "workload")) return .workload;
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
        \\  zug agent [--node id] [--profile profile] [--listen ip:port] [--once] [--mock-gpu]
        \\  zug upload <workload/> --agent http://host:port [--deploy] [--chunk-size bytes]
        \\  zug wit <descriptor> [--out file.wit]
        \\  zug run <workload/> [--profile profile] [--arg i32] [--stdin text] [--json] [--mock-gpu]
        \\  zug check <file.onnx|file.wasm|workload/> [--kind auto|onnx|wasm|workload] [--manifest manifest] [--model model.onnx] [--export name] [--memory bytes] [--json]
        \\  zug inspect <model.onnx>
        \\  zug bench <model.onnx> --input name=file.f32 [--warmup count] [--iterations count] [--format text|json] [--trace]
        \\  zug <model.onnx> [--input name=file.f32] [--output name=file.raw] [--expect name=file.raw] [--tolerance value]
        \\  zug wasm <module.wasm> [--manifest manifest] [--export name] [--model model.onnx] [--model-offset bytes] [--arg i32] [--mock-gpu]
        \\
        \\onnx:
        \\  scope prints the current runtime capability contract
        \\  check reports compatibility against the current runtime scope
        \\  --output writes a named graph output as raw little-endian tensor bytes
        \\  --expect compares a named graph output against raw little-endian tensor bytes
        \\  --tolerance sets max absolute difference for float32 expectations
        \\  bench uses a non-debug counting allocator and reports timing plus allocation pressure
        \\
        \\workload:
        \\  run is the central local workflow for a WASM guest plus model package
        \\  --profile overrides the manifest target profile for local execution
        \\  --arg passes an i32 argument to the configured workload entrypoint
        \\  --stdin provides guest stdin bytes through WASI fd_read
        \\  --json emits machine-readable execution output
        \\  --mock-gpu enables a host mock GPU device and mock accelerator backend catalog
        \\
        \\wasm:
        \\  --manifest checks runtime requirements before execution
        \\  --model exposes a host-managed model to zug_nn.load_preloaded_graph
        \\  --model-offset also writes that model into guest memory for legacy guests
        \\  --mock-gpu enables a host mock GPU device and mock accelerator backend catalog
        \\
        \\upload:
        \\  uploads zug.toml, the WASM module, and optional model into the agent artifact store
        \\  --deploy checks and registers the uploaded workload on the target agent
        \\
        \\wit:
        \\  emits WebAssembly Component Model WIT from a Zig descriptor
        \\  descriptors: edge-inference, component-smoke, src/component/edge_inference_wit.zig, src/component/component_smoke_wit.zig
        \\  --out writes to a file instead of printing the WIT text
        \\
        \\agent:
        \\  serves GET /health, GET /capabilities, GET /activity, GET /workloads, GET /telemetry over HTTP
        \\  POST /workloads/check with {{"path":"workload/"}} returns compatibility JSON
        \\  POST /workloads/deploy with {{"path":"workload/"}} registers a supported workload
        \\  POST /workloads/<id>/invoke queues an async invoke job with optional {{"args":[i32],"stdin":"..."}}
        \\  GET /jobs and GET /jobs/<id> report async workload invocation job status
        \\  POST /jobs/<id>/cancel requests cancellation for a queued or running invoke job
        \\  POST /packages/files with {{"path":"workloads/demo/model.onnx","offset":0,"data_hex":"...","truncate":true}} uploads package/model chunks
        \\  GET /packages lists uploaded package files and stored paths
        \\  POST /telemetry/events with {{"source":"camera","kind":"frame_drop","severity":"warn","message":"..."}} records an event
        \\  POST /telemetry/metrics with {{"source":"runtime","name":"latency_ms","value":14.5,"unit":"ms"}} records a scalar metric
        \\  POST /telemetry/traces records a local trace span
        \\  POST /v1/logs, POST /v1/metrics and POST /v1/traces accept OTLP/HTTP JSON from OpenTelemetry clients
        \\  --once accepts one TCP request and then exits
        \\  --mock-gpu advertises a mock integrated GPU device and accelerator backends to deployed WASM workloads
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

fn runWorkload(allocator: std.mem.Allocator, cli: *const CliArgs) !void {
    var result = try workload_runner.run(allocator, .{
        .path = cli.model_path,
        .profile_name = cli.run_profile_name,
        .args = cli.wasm_args.items,
        .stdin = cli.wasm_stdin orelse &.{},
        .gpu = if (cli.mock_gpu) gpu.mockEdgeCapabilities() else .{},
        .accelerators = if (cli.mock_gpu) accelerator.mockEdgeCapabilities() else accelerator.defaultCapabilities(),
    });
    defer result.deinit(allocator);

    switch (cli.check_output_format) {
        .text => printWorkloadRunText(result),
        .json => try printWorkloadRunJson(allocator, result),
    }
}

const UploadEndpoint = struct {
    host: []const u8,
    port: u16,

    fn deinit(self: *UploadEndpoint, allocator: std.mem.Allocator) void {
        allocator.free(self.host);
        self.* = undefined;
    }
};

const UploadPackage = struct {
    name: []const u8,
    remote_dir: []const u8,
    deploy_path: []const u8,
    manifest_bytes: []const u8,
    wasm_remote_name: []const u8,
    model_remote_name: ?[]const u8,

    fn deinit(self: *UploadPackage, allocator: std.mem.Allocator) void {
        allocator.free(self.name);
        allocator.free(self.remote_dir);
        allocator.free(self.deploy_path);
        allocator.free(self.manifest_bytes);
        allocator.free(self.wasm_remote_name);
        if (self.model_remote_name) |name| allocator.free(name);
        self.* = undefined;
    }
};

fn runUpload(allocator: std.mem.Allocator, cli: *const CliArgs) !void {
    var loaded = try workload.Loaded.load(allocator, cli.model_path);
    defer loaded.deinit(allocator);

    var endpoint = try parseUploadEndpoint(allocator, cli.upload_agent_url.?);
    defer endpoint.deinit(allocator);

    var package = try buildUploadPackage(allocator, loaded);
    defer package.deinit(allocator);

    std.debug.print("zug upload workload\n", .{});
    std.debug.print("  name: {s}\n", .{package.name});
    std.debug.print("  agent: http://{s}:{d}\n", .{ endpoint.host, endpoint.port });
    std.debug.print("  package: {s}\n", .{package.remote_dir});

    const manifest_remote_path = try joinRemotePath(allocator, package.remote_dir, workload.manifest_file_name);
    defer allocator.free(manifest_remote_path);
    try uploadBytes(allocator, endpoint, manifest_remote_path, package.manifest_bytes, cli.upload_chunk_size);
    std.debug.print("  uploaded: {s} ({d} bytes)\n", .{ manifest_remote_path, package.manifest_bytes.len });

    const wasm_bytes = try readUploadFile(allocator, loaded.wasm_path, max_wasm_bytes);
    defer allocator.free(wasm_bytes);
    const wasm_remote_path = try joinRemotePath(allocator, package.remote_dir, package.wasm_remote_name);
    defer allocator.free(wasm_remote_path);
    try uploadBytes(allocator, endpoint, wasm_remote_path, wasm_bytes, cli.upload_chunk_size);
    std.debug.print("  uploaded: {s} ({d} bytes)\n", .{ wasm_remote_path, wasm_bytes.len });

    if (loaded.model_path) |model_path| {
        const model_remote_name = package.model_remote_name orelse return error.MissingUploadModelName;
        const model_bytes = try readUploadFile(allocator, model_path, max_model_bytes);
        defer allocator.free(model_bytes);
        const model_remote_path = try joinRemotePath(allocator, package.remote_dir, model_remote_name);
        defer allocator.free(model_remote_path);
        try uploadBytes(allocator, endpoint, model_remote_path, model_bytes, cli.upload_chunk_size);
        std.debug.print("  uploaded: {s} ({d} bytes)\n", .{ model_remote_path, model_bytes.len });
    }

    if (cli.upload_deploy) {
        const check_response = try postPathRequest(allocator, endpoint, "/workloads/check", package.deploy_path);
        defer allocator.free(check_response);
        const deploy_response = try postPathRequest(allocator, endpoint, "/workloads/deploy", package.deploy_path);
        defer allocator.free(deploy_response);
        std.debug.print("  deploy: {s}\n", .{trimResponseForDisplay(deploy_response)});
    } else {
        std.debug.print("  deploy: skipped (use --deploy to register {s})\n", .{package.deploy_path});
    }
}

fn parseUploadEndpoint(allocator: std.mem.Allocator, raw_url: []const u8) !UploadEndpoint {
    var rest = std.mem.trim(u8, raw_url, " \t\r\n");
    if (rest.len == 0) return error.InvalidUploadAgent;

    if (std.mem.startsWith(u8, rest, "https://")) return error.UploadHttpsUnsupported;
    if (std.mem.startsWith(u8, rest, "http://")) rest = rest["http://".len..];

    const slash_index = std.mem.indexOfScalar(u8, rest, '/');
    const host_port = if (slash_index) |index| rest[0..index] else rest;
    if (host_port.len == 0) return error.InvalidUploadAgent;

    if (slash_index) |index| {
        const path = rest[index..];
        if (!std.mem.eql(u8, path, "/")) return error.UploadAgentPathUnsupported;
    }

    if (host_port[0] == '[') return error.UploadIpv6Unsupported;
    const colon_index = std.mem.lastIndexOfScalar(u8, host_port, ':') orelse return error.MissingUploadAgentPort;
    if (colon_index == 0 or colon_index + 1 >= host_port.len) return error.InvalidUploadAgent;

    const host = host_port[0..colon_index];
    const port = try std.fmt.parseInt(u16, host_port[colon_index + 1 ..], 10);

    return .{
        .host = try allocator.dupe(u8, host),
        .port = port,
    };
}

fn buildUploadPackage(allocator: std.mem.Allocator, loaded: workload.Loaded) !UploadPackage {
    const base_name = loaded.manifest.name orelse std.fs.path.basename(loaded.root_path);
    const safe_name = try sanitizeUploadName(allocator, base_name);
    errdefer allocator.free(safe_name);

    const remote_dir = try std.fmt.allocPrint(allocator, "workloads/{s}", .{safe_name});
    errdefer allocator.free(remote_dir);

    const deploy_path = try std.fmt.allocPrint(allocator, ".zug-artifacts/{s}", .{remote_dir});
    errdefer allocator.free(deploy_path);

    const wasm_remote_name = try sanitizeUploadName(allocator, std.fs.path.basename(loaded.wasm_path));
    errdefer allocator.free(wasm_remote_name);

    const model_remote_name = if (loaded.model_path) |path| try sanitizeUploadName(allocator, std.fs.path.basename(path)) else null;
    errdefer if (model_remote_name) |name| allocator.free(name);

    const manifest_bytes = try renderUploadManifest(allocator, loaded.manifest, safe_name, wasm_remote_name, model_remote_name);
    errdefer allocator.free(manifest_bytes);

    return .{
        .name = safe_name,
        .remote_dir = remote_dir,
        .deploy_path = deploy_path,
        .manifest_bytes = manifest_bytes,
        .wasm_remote_name = wasm_remote_name,
        .model_remote_name = model_remote_name,
    };
}

fn sanitizeUploadName(allocator: std.mem.Allocator, name: []const u8) ![]const u8 {
    const trimmed = std.mem.trim(u8, name, " \t\r\n");
    if (trimmed.len == 0 or std.mem.eql(u8, trimmed, ".") or std.mem.eql(u8, trimmed, "..")) {
        return error.InvalidUploadName;
    }

    const out = try allocator.alloc(u8, trimmed.len);
    errdefer allocator.free(out);
    var wrote_valid = false;
    for (trimmed, 0..) |char, index| {
        out[index] = switch (char) {
            'a'...'z', 'A'...'Z', '0'...'9', '.', '_', '-' => char,
            else => '-',
        };
        if (out[index] != '.' and out[index] != '-') wrote_valid = true;
    }

    if (!wrote_valid) return error.InvalidUploadName;
    if (std.mem.indexOf(u8, out, "..") != null) return error.InvalidUploadName;
    return out;
}

fn renderUploadManifest(
    allocator: std.mem.Allocator,
    manifest: workload.Manifest,
    name: []const u8,
    wasm_remote_name: []const u8,
    model_remote_name: ?[]const u8,
) ![]u8 {
    var out: std.Io.Writer.Allocating = .init(allocator);
    errdefer out.deinit();

    try out.writer.writeAll("name = ");
    try writeTomlString(&out.writer, name);
    try out.writer.writeByte('\n');

    try out.writer.writeAll("entrypoint = ");
    try writeTomlString(&out.writer, manifest.entrypointName());
    try out.writer.writeByte('\n');

    try out.writer.writeAll("wasm = ");
    try writeTomlString(&out.writer, wasm_remote_name);
    try out.writer.writeByte('\n');

    if (model_remote_name) |model_name| {
        try out.writer.writeAll("model = ");
        try writeTomlString(&out.writer, model_name);
        try out.writer.writeByte('\n');
    }

    if (manifest.model_encoding) |encoding| {
        try out.writer.writeAll("model_encoding = ");
        try writeTomlString(&out.writer, encoding);
        try out.writer.writeByte('\n');
    }

    try out.writer.writeAll("\n[requires]\n");
    try out.writer.print("wasi_nn = {}\n", .{manifest.requires.wasi_nn});
    try out.writer.print("wasi_http = {}\n", .{manifest.requires.wasi_http});
    if (manifest.requires.memory_bytes) |memory_bytes| try out.writer.print("memory_bytes = {d}\n", .{memory_bytes});
    if (manifest.requires.dtype) |dtype| {
        try out.writer.writeAll("dtype = ");
        try writeTomlString(&out.writer, dtypeName(dtype));
        try out.writer.writeByte('\n');
    }
    if (manifest.requires.target_name) |target_name| {
        try out.writer.writeAll("target = ");
        try writeTomlString(&out.writer, target_name);
        try out.writer.writeByte('\n');
    }

    if (manifest.network.input != null or
        manifest.network.output != null or
        manifest.network.ingress.items.len != 0 or
        manifest.network.egress.items.len != 0 or
        manifest.network.max_inflight != null or
        manifest.network.request_timeout_ms != null or
        manifest.network.public_egress)
    {
        try out.writer.writeAll("\n[network]\n");
        if (manifest.network.input) |protocol| try writeProtocolField(&out.writer, "input", protocol);
        if (manifest.network.output) |protocol| try writeProtocolField(&out.writer, "output", protocol);
        for (manifest.network.ingress.items) |protocol| try writeProtocolField(&out.writer, "ingress", protocol);
        for (manifest.network.egress.items) |protocol| try writeProtocolField(&out.writer, "egress", protocol);
        if (manifest.network.max_inflight) |max_inflight| try out.writer.print("max_inflight = {d}\n", .{max_inflight});
        if (manifest.network.request_timeout_ms) |timeout| try out.writer.print("request_timeout_ms = {d}\n", .{timeout});
        if (manifest.network.public_egress) try out.writer.writeAll("public_egress = true\n");
    }

    return try out.toOwnedSlice();
}

fn writeTomlString(writer: *std.Io.Writer, value: []const u8) !void {
    try writer.writeByte('"');
    for (value) |char| {
        switch (char) {
            '"' => try writer.writeAll("\\\""),
            '\\' => try writer.writeAll("\\\\"),
            '\n' => try writer.writeAll("\\n"),
            '\r' => try writer.writeAll("\\r"),
            '\t' => try writer.writeAll("\\t"),
            else => {
                if (char < 32) return error.InvalidTomlString;
                try writer.writeByte(char);
            },
        }
    }
    try writer.writeByte('"');
}

fn writeProtocolField(writer: *std.Io.Writer, key: []const u8, protocol: network.Protocol) !void {
    try writer.print("{s} = ", .{key});
    try writeTomlString(writer, network.protocolName(protocol));
    try writer.writeByte('\n');
}

fn dtypeName(dtype: target_profile.DType) []const u8 {
    return switch (dtype) {
        .float32 => "float32",
        .int64 => "int64",
        .int32 => "int32",
        .uint8 => "uint8",
        .bool => "bool",
    };
}

fn joinRemotePath(allocator: std.mem.Allocator, dir: []const u8, name: []const u8) ![]const u8 {
    if (dir.len == 0 or name.len == 0) return error.InvalidUploadPath;
    return std.fmt.allocPrint(allocator, "{s}/{s}", .{ dir, name });
}

fn readUploadFile(allocator: std.mem.Allocator, path: []const u8, limit: usize) ![]u8 {
    return try std.Io.Dir.cwd().readFileAlloc(
        std.Options.debug_io,
        path,
        allocator,
        .limited(limit),
    );
}

fn uploadBytes(
    allocator: std.mem.Allocator,
    endpoint: UploadEndpoint,
    remote_path: []const u8,
    bytes: []const u8,
    chunk_size: usize,
) !void {
    if (chunk_size == 0) return error.InvalidUploadChunkSize;

    if (bytes.len == 0) {
        try uploadChunk(allocator, endpoint, remote_path, 0, &.{}, true);
        return;
    }

    var offset: usize = 0;
    while (offset < bytes.len) {
        const end = @min(bytes.len, offset + chunk_size);
        try uploadChunk(allocator, endpoint, remote_path, offset, bytes[offset..end], offset == 0);
        offset = end;
    }
}

fn uploadChunk(
    allocator: std.mem.Allocator,
    endpoint: UploadEndpoint,
    remote_path: []const u8,
    offset: usize,
    chunk: []const u8,
    truncate: bool,
) !void {
    const body = try buildUploadChunkBody(allocator, remote_path, offset, chunk, truncate);
    defer allocator.free(body);

    const response = try httpPostJson(allocator, endpoint, "/packages/files", body);
    defer allocator.free(response);
}

fn buildUploadChunkBody(
    allocator: std.mem.Allocator,
    remote_path: []const u8,
    offset: usize,
    chunk: []const u8,
    truncate: bool,
) ![]u8 {
    const hex = try hexEncodeAlloc(allocator, chunk);
    defer allocator.free(hex);

    var out: std.Io.Writer.Allocating = .init(allocator);
    errdefer out.deinit();

    var json: std.json.Stringify = .{ .writer = &out.writer };
    try json.beginObject();
    try json.objectField("path");
    try json.write(remote_path);
    try json.objectField("offset");
    try json.write(offset);
    try json.objectField("data_hex");
    try json.write(hex);
    try json.objectField("truncate");
    try json.write(truncate);
    try json.endObject();

    return try out.toOwnedSlice();
}

fn postPathRequest(
    allocator: std.mem.Allocator,
    endpoint: UploadEndpoint,
    target: []const u8,
    path: []const u8,
) ![]u8 {
    var out: std.Io.Writer.Allocating = .init(allocator);
    errdefer out.deinit();

    var json: std.json.Stringify = .{ .writer = &out.writer };
    try json.beginObject();
    try json.objectField("path");
    try json.write(path);
    try json.endObject();

    const body = try out.toOwnedSlice();
    defer allocator.free(body);

    return try httpPostJson(allocator, endpoint, target, body);
}

fn hexEncodeAlloc(allocator: std.mem.Allocator, bytes: []const u8) ![]u8 {
    const out = try allocator.alloc(u8, bytes.len * 2);
    for (bytes, 0..) |byte, index| {
        const high = byte >> 4;
        const low = byte & 0x0f;
        out[index * 2] = hexDigit(high);
        out[index * 2 + 1] = hexDigit(low);
    }
    return out;
}

fn hexDigit(value: u8) u8 {
    return if (value < 10) '0' + value else 'a' + (value - 10);
}

fn httpPostJson(
    allocator: std.mem.Allocator,
    endpoint: UploadEndpoint,
    target: []const u8,
    body: []const u8,
) ![]u8 {
    const response = try httpRequest(allocator, endpoint, "POST", target, "application/json", body);
    errdefer allocator.free(response);
    ensureHttpOk(response) catch |err| {
        std.debug.print("  agent response: {s}\n", .{trimResponseForDisplay(response)});
        return err;
    };
    return try responseBodyOwned(allocator, response);
}

fn httpRequest(
    allocator: std.mem.Allocator,
    endpoint: UploadEndpoint,
    method: []const u8,
    target: []const u8,
    content_type: []const u8,
    body: []const u8,
) ![]u8 {
    const io = std.Options.debug_io;
    var address = try resolveUploadAddress(io, endpoint);
    var stream = try address.connect(io, .{ .mode = .stream });
    defer stream.close(io);

    var write_buffer: [4096]u8 = undefined;
    var writer_state = stream.writer(io, &write_buffer);
    try writer_state.interface.print(
        "{s} {s} HTTP/1.1\r\nHost: {s}:{d}\r\nContent-Type: {s}\r\nContent-Length: {d}\r\nConnection: close\r\n\r\n",
        .{ method, target, endpoint.host, endpoint.port, content_type, body.len },
    );
    try writer_state.interface.writeAll(body);
    try writer_state.interface.flush();

    var read_buffer: [4096]u8 = undefined;
    var reader_state = stream.reader(io, &read_buffer);
    var response: std.Io.Writer.Allocating = .init(allocator);
    errdefer response.deinit();

    var total: usize = 0;
    var chunk: [4096]u8 = undefined;
    while (true) {
        const read_count = try reader_state.interface.readSliceShort(&chunk);
        if (read_count == 0) break;
        total = try std.math.add(usize, total, read_count);
        if (total > max_upload_response_bytes) return error.UploadHttpResponseTooLarge;
        try response.writer.writeAll(chunk[0..read_count]);
    }

    return try response.toOwnedSlice();
}

fn resolveUploadAddress(io: std.Io, endpoint: UploadEndpoint) !net.IpAddress {
    if (std.mem.eql(u8, endpoint.host, "localhost")) return .{ .ip4 = .loopback(endpoint.port) };
    if (std.mem.eql(u8, endpoint.host, "127.0.0.1")) return .{ .ip4 = .loopback(endpoint.port) };
    return try net.IpAddress.resolve(io, endpoint.host, endpoint.port);
}

fn ensureHttpOk(response: []const u8) !void {
    if (std.mem.startsWith(u8, response, "HTTP/1.1 200 ") or
        std.mem.startsWith(u8, response, "HTTP/1.0 200 "))
    {
        return;
    }
    return error.UploadHttpFailed;
}

fn responseBodyOwned(allocator: std.mem.Allocator, response: []u8) ![]u8 {
    const body_start = if (std.mem.indexOf(u8, response, "\r\n\r\n")) |index|
        index + 4
    else if (std.mem.indexOf(u8, response, "\n\n")) |index|
        index + 2
    else
        return error.InvalidHttpResponse;

    const body = try allocator.dupe(u8, response[body_start..]);
    allocator.free(response);
    return body;
}

fn trimResponseForDisplay(response_body: []const u8) []const u8 {
    const trimmed = std.mem.trim(u8, response_body, " \t\r\n");
    if (trimmed.len <= 240) return trimmed;
    return trimmed[0..240];
}

fn printWorkloadRunText(result: workload_runner.Result) void {
    std.debug.print("zug run workload\n", .{});
    std.debug.print("  path: {s}\n", .{result.path});
    std.debug.print("  name: {s}\n", .{result.name});
    std.debug.print("  profile: {s}\n", .{result.profile});
    std.debug.print("  entrypoint: {s}\n", .{result.entrypoint});
    std.debug.print("  wasm: {s} ({d} bytes)\n", .{ result.wasm_path, result.wasm_bytes });
    if (result.model_path.len != 0) {
        std.debug.print("  model: {s} ({d} bytes)\n", .{ result.model_path, result.model_bytes });
    } else {
        std.debug.print("  model: <none>\n", .{});
    }
    std.debug.print("  memory_bytes: {d}\n", .{result.memory_bytes});
    std.debug.print("  duration_ns: {d}\n", .{result.duration_ns});

    if (result.stdout.len != 0) {
        std.debug.print("stdout:\n{s}", .{result.stdout});
        if (result.stdout[result.stdout.len - 1] != '\n') std.debug.print("\n", .{});
    }
    if (result.stderr.len != 0) {
        std.debug.print("stderr:\n{s}", .{result.stderr});
        if (result.stderr[result.stderr.len - 1] != '\n') std.debug.print("\n", .{});
    }
    if (result.exit_code) |exit_code| {
        std.debug.print("wasm exit: {d}\n", .{exit_code});
    }

    printNamedWasmResult("workload result", result.value);
}

fn printWorkloadRunJson(allocator: std.mem.Allocator, result: workload_runner.Result) !void {
    var out: std.Io.Writer.Allocating = .init(allocator);
    defer out.deinit();

    var json: std.json.Stringify = .{ .writer = &out.writer };
    try json.beginObject();
    try json.objectField("kind");
    try json.write("workload_run");
    try json.objectField("status");
    try json.write("completed");
    try json.objectField("path");
    try json.write(result.path);
    try json.objectField("name");
    try json.write(result.name);
    try json.objectField("profile");
    try json.write(result.profile);
    try json.objectField("entrypoint");
    try json.write(result.entrypoint);
    try json.objectField("wasm");
    try json.write(result.wasm_path);
    try json.objectField("wasm_bytes");
    try json.write(result.wasm_bytes);
    try json.objectField("model");
    try json.write(result.model_path);
    try json.objectField("model_bytes");
    try json.write(result.model_bytes);
    try json.objectField("memory_bytes");
    try json.write(result.memory_bytes);
    try json.objectField("duration_ns");
    try json.write(result.duration_ns);
    try json.objectField("exit_code_present");
    try json.write(result.exit_code != null);
    try json.objectField("exit_code");
    try json.write(result.exit_code orelse 0);
    try json.objectField("stdout");
    try json.write(result.stdout);
    try json.objectField("stderr");
    try json.write(result.stderr);
    try json.objectField("result");
    try writeWasmValueJson(&json, result.value);
    try json.endObject();

    const body = try out.toOwnedSlice();
    defer allocator.free(body);
    std.debug.print("{s}\n", .{body});
}

fn writeWasmValueJson(json: *std.json.Stringify, value: ?wasm_interpreter.Value) !void {
    try json.beginObject();
    if (value) |actual| {
        switch (actual) {
            .i32 => |payload| {
                try json.objectField("type");
                try json.write("i32");
                try json.objectField("value");
                try json.write(@as(i32, @bitCast(payload)));
            },
            .i64 => |payload| {
                try json.objectField("type");
                try json.write("i64");
                try json.objectField("value");
                try json.write(@as(i64, @bitCast(payload)));
            },
            .f32 => |payload| {
                try json.objectField("type");
                try json.write("f32");
                try json.objectField("value");
                try json.write(payload);
            },
            .f64 => |payload| {
                try json.objectField("type");
                try json.write("f64");
                try json.objectField("value");
                try json.write(payload);
            },
            .funcref => |payload| {
                try json.objectField("type");
                try json.write("funcref");
                try json.objectField("value");
                try json.write(payload orelse 0);
                try json.objectField("null");
                try json.write(payload == null);
            },
            .v128 => |payload| {
                try json.objectField("type");
                try json.write("v128");
                try json.objectField("bytes");
                try json.beginArray();
                for (payload) |byte| {
                    try json.write(byte);
                }
                try json.endArray();
            },
        }
    } else {
        try json.objectField("type");
        try json.write("none");
    }
    try json.endObject();
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

    var host = wasi_nn_abi.Host.initWithAccelerators(
        allocator,
        if (cli.mock_gpu) accelerator.mockEdgeCapabilities() else accelerator.defaultCapabilities(),
    );
    defer host.deinit();

    var surface = wasi_nn_abi.Surface.init(allocator, &host, &wasm_instance.memory);
    surface.setPreloadedModel(model_bytes);

    var resolver = wasm_imports.Resolver.init(&surface);
    resolver.gpu = if (cli.mock_gpu) gpu.mockEdgeCapabilities() else .{};
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
    printNamedWasmResult("wasm result", result);
}

fn printNamedWasmResult(label: []const u8, result: ?wasm_interpreter.Value) void {
    std.debug.print("{s}: ", .{label});

    if (result) |value| {
        switch (value) {
            .i32 => |actual| std.debug.print("i32 {d}", .{@as(i32, @bitCast(actual))}),
            .i64 => |actual| std.debug.print("i64 {d}", .{@as(i64, @bitCast(actual))}),
            .f32 => |actual| std.debug.print("f32 {d}", .{actual}),
            .f64 => |actual| std.debug.print("f64 {d}", .{actual}),
            .funcref => |actual| {
                if (actual) |function_index| {
                    std.debug.print("funcref {d}", .{function_index});
                } else {
                    std.debug.print("funcref null", .{});
                }
            },
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
