const std = @import("std");
const protobuf = @import("protobuf");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const exe = b.addExecutable(.{
        .name = "zug",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/cli.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });

    const protobuf_dep = b.dependency("protobuf", .{
        .target = target,
        .optimize = optimize,
    });

    exe.root_module.addImport("protobuf", protobuf_dep.module("protobuf"));

    b.installArtifact(exe);

    const run_exe = b.addRunArtifact(exe);

    if (b.args) |args| {
        run_exe.addArgs(args);
    }

    const run_step = b.step("run", "Run the application");
    run_step.dependOn(&run_exe.step);

    const bench_mobilenet = b.addRunArtifact(exe);
    bench_mobilenet.addArgs(&.{
        "bench",
        "models/mobilenetv2-12.onnx",
        "--input",
        "input=tmp/mobilenetv2-zero.f32",
        "--warmup",
        "1",
        "--iterations",
        "3",
    });

    const bench_mobilenet_step = b.step("bench-mobilenet", "Benchmark MobileNetV2 ONNX execution");
    bench_mobilenet_step.dependOn(&bench_mobilenet.step);

    const bench_mobilenet_json = b.addRunArtifact(exe);
    bench_mobilenet_json.addArgs(&.{
        "bench",
        "models/mobilenetv2-12.onnx",
        "--input",
        "input=tmp/mobilenetv2-zero.f32",
        "--warmup",
        "1",
        "--iterations",
        "3",
        "--format",
        "json",
    });

    const bench_mobilenet_json_step = b.step("bench-mobilenet-json", "Benchmark MobileNetV2 ONNX execution as JSON");
    bench_mobilenet_json_step.dependOn(&bench_mobilenet_json.step);

    const wasi_nn_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/wasi_nn.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });

    wasi_nn_tests.root_module.addImport("protobuf", protobuf_dep.module("protobuf"));

    const wasi_nn_abi_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/wasi_nn_abi.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });

    wasi_nn_abi_tests.root_module.addImport("protobuf", protobuf_dep.module("protobuf"));

    const session_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/session.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });

    session_tests.root_module.addImport("protobuf", protobuf_dep.module("protobuf"));

    const benchmark_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/benchmark.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });

    const accelerator_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/accelerator.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });

    const check_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/check.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });

    check_tests.root_module.addImport("protobuf", protobuf_dep.module("protobuf"));

    const scope_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/scope.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });

    scope_tests.root_module.addImport("protobuf", protobuf_dep.module("protobuf"));

    const target_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/target.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });

    const network_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/network.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });

    const telemetry_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/telemetry.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });

    const control_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/control.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });

    const package_transfer_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/package_transfer.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });

    const gpu_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/gpu.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });

    const agent_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/agent.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });

    agent_tests.root_module.addImport("protobuf", protobuf_dep.module("protobuf"));

    const workload_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/workload.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });

    const workload_runner_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/workload_runner.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });

    workload_runner_tests.root_module.addImport("protobuf", protobuf_dep.module("protobuf"));

    const component_wit_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/component/wit.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });

    const wit_generator_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/wit_generator.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });

    const wasi_nn_abi_module = b.createModule(.{
        .root_source_file = b.path("src/wasi_nn_abi.zig"),
        .target = target,
        .optimize = optimize,
    });

    wasi_nn_abi_module.addImport("protobuf", protobuf_dep.module("protobuf"));

    const wasm_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/wasm_tests.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });

    wasm_tests.root_module.addImport("protobuf", protobuf_dep.module("protobuf"));

    const component_smoke_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/component_smoke_tests.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });

    component_smoke_tests.root_module.addImport("protobuf", protobuf_dep.module("protobuf"));

    const deployment_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/deployment_tests.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });

    deployment_tests.root_module.addImport("protobuf", protobuf_dep.module("protobuf"));

    const run_wasi_nn_tests = b.addRunArtifact(wasi_nn_tests);
    const run_wasi_nn_abi_tests = b.addRunArtifact(wasi_nn_abi_tests);
    const run_session_tests = b.addRunArtifact(session_tests);
    const run_benchmark_tests = b.addRunArtifact(benchmark_tests);
    const run_accelerator_tests = b.addRunArtifact(accelerator_tests);
    const run_check_tests = b.addRunArtifact(check_tests);
    const run_scope_tests = b.addRunArtifact(scope_tests);
    const run_target_tests = b.addRunArtifact(target_tests);
    const run_network_tests = b.addRunArtifact(network_tests);
    const run_telemetry_tests = b.addRunArtifact(telemetry_tests);
    const run_control_tests = b.addRunArtifact(control_tests);
    const run_package_transfer_tests = b.addRunArtifact(package_transfer_tests);
    const run_gpu_tests = b.addRunArtifact(gpu_tests);
    const run_agent_tests = b.addRunArtifact(agent_tests);
    const run_workload_tests = b.addRunArtifact(workload_tests);
    const run_workload_runner_tests = b.addRunArtifact(workload_runner_tests);
    const run_component_wit_tests = b.addRunArtifact(component_wit_tests);
    const run_wit_generator_tests = b.addRunArtifact(wit_generator_tests);
    const run_wasm_tests = b.addRunArtifact(wasm_tests);
    const run_component_smoke_tests = b.addRunArtifact(component_smoke_tests);
    const run_deployment_tests = b.addRunArtifact(deployment_tests);

    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_wasi_nn_tests.step);
    test_step.dependOn(&run_wasi_nn_abi_tests.step);
    test_step.dependOn(&run_session_tests.step);
    test_step.dependOn(&run_benchmark_tests.step);
    test_step.dependOn(&run_accelerator_tests.step);
    test_step.dependOn(&run_check_tests.step);
    test_step.dependOn(&run_scope_tests.step);
    test_step.dependOn(&run_target_tests.step);
    test_step.dependOn(&run_network_tests.step);
    test_step.dependOn(&run_telemetry_tests.step);
    test_step.dependOn(&run_control_tests.step);
    test_step.dependOn(&run_package_transfer_tests.step);
    test_step.dependOn(&run_gpu_tests.step);
    test_step.dependOn(&run_agent_tests.step);
    test_step.dependOn(&run_workload_tests.step);
    test_step.dependOn(&run_workload_runner_tests.step);
    test_step.dependOn(&run_component_wit_tests.step);
    test_step.dependOn(&run_wit_generator_tests.step);
    test_step.dependOn(&run_wasm_tests.step);
    test_step.dependOn(&run_deployment_tests.step);

    const test_wasi_nn_step = b.step("test-wasi-nn", "Run WASI-NN tests");
    test_wasi_nn_step.dependOn(&run_wasi_nn_tests.step);
    test_wasi_nn_step.dependOn(&run_wasi_nn_abi_tests.step);

    const test_wasi_nn_abi_step = b.step("test-wasi-nn-abi", "Run WASI-NN ABI tests");
    test_wasi_nn_abi_step.dependOn(&run_wasi_nn_abi_tests.step);

    const test_wasm_step = b.step("test-wasm", "Run WASM runtime skeleton tests");
    test_wasm_step.dependOn(&run_wasm_tests.step);

    const test_deployment_step = b.step("test-deployment", "Run edge deployment environment tests");
    test_deployment_step.dependOn(&run_deployment_tests.step);

    const guest_basic = b.addSystemCommand(&.{
        b.graph.zig_exe,
        "build-exe",
        "guests/basic.zig",
        "-target",
        "wasm32-freestanding",
        "-O",
        "ReleaseSmall",
        "-fno-entry",
        "-rdynamic",
        "-femit-bin=zig-out/basic.wasm",
    });

    const guest_step = b.step("guest-basic", "Build the basic external wasm guest");
    guest_step.dependOn(&guest_basic.step);

    const guest_component_smoke = b.addSystemCommand(&.{
        b.graph.zig_exe,
        "build-exe",
        "guests/component_smoke.zig",
        "-target",
        "wasm32-freestanding",
        "-O",
        "ReleaseSmall",
        "-fno-entry",
        "-rdynamic",
        "-femit-bin=zig-out/component_smoke.wasm",
    });

    const guest_component_smoke_step = b.step("guest-component-smoke", "Build the component smoke external wasm guest");
    guest_component_smoke_step.dependOn(&guest_component_smoke.step);

    const guest_wasi_log = b.addSystemCommand(&.{
        b.graph.zig_exe,
        "build-exe",
        "guests/wasi_log.zig",
        "-target",
        "wasm32-freestanding",
        "-O",
        "ReleaseSmall",
        "-fno-entry",
        "-rdynamic",
        "-femit-bin=zig-out/wasi_log.wasm",
    });

    const guest_wasi_log_step = b.step("guest-wasi-log", "Build the WASI fd_write external wasm guest");
    guest_wasi_log_step.dependOn(&guest_wasi_log.step);

    const guest_wasi_nn_smoke = b.addSystemCommand(&.{
        b.graph.zig_exe,
        "build-exe",
        "guests/wasi_nn_smoke.zig",
        "-target",
        "wasm32-freestanding",
        "-O",
        "ReleaseSmall",
        "-fno-entry",
        "-rdynamic",
        "-femit-bin=zig-out/wasi_nn_smoke.wasm",
    });

    const guest_wasi_nn_smoke_step = b.step("guest-wasi-nn-smoke", "Build the WASI-NN external wasm smoke guest");
    guest_wasi_nn_smoke_step.dependOn(&guest_wasi_nn_smoke.step);

    const guest_wasi_nn_full = b.addSystemCommand(&.{
        b.graph.zig_exe,
        "build-exe",
        "guests/wasi_nn_full.zig",
        "-target",
        "wasm32-freestanding",
        "-O",
        "ReleaseSmall",
        "-fno-entry",
        "-rdynamic",
        "-femit-bin=zig-out/wasi_nn_full.wasm",
    });

    const guest_wasi_nn_full_step = b.step("guest-wasi-nn-full", "Build the full WASI-NN external wasm guest");
    guest_wasi_nn_full_step.dependOn(&guest_wasi_nn_full.step);

    const guest_gpu_probe = b.addSystemCommand(&.{
        b.graph.zig_exe,
        "build-exe",
        "guests/gpu_probe.zig",
        "-target",
        "wasm32-freestanding",
        "-O",
        "ReleaseSmall",
        "-fno-entry",
        "-rdynamic",
        "-femit-bin=zig-out/gpu_probe.wasm",
    });

    const guest_gpu_probe_step = b.step("guest-gpu-probe", "Build the GPU control external wasm guest");
    guest_gpu_probe_step.dependOn(&guest_gpu_probe.step);

    const guests_step = b.step("guests", "Build all external wasm guest fixtures");
    guests_step.dependOn(&guest_basic.step);
    guests_step.dependOn(&guest_component_smoke.step);
    guests_step.dependOn(&guest_wasi_log.step);
    guests_step.dependOn(&guest_wasi_nn_smoke.step);
    guests_step.dependOn(&guest_wasi_nn_full.step);
    guests_step.dependOn(&guest_gpu_probe.step);

    const gen_component_smoke_wit_run = b.addRunArtifact(exe);
    gen_component_smoke_wit_run.addArg("wit");
    gen_component_smoke_wit_run.addArg("component-smoke");
    gen_component_smoke_wit_run.addArg("--out");
    gen_component_smoke_wit_run.addArg("zig-out/component-smoke.wit");

    const gen_component_smoke_wit = b.step("gen-component-smoke-wit", "Generate the component smoke WIT file");
    gen_component_smoke_wit.dependOn(&gen_component_smoke_wit_run.step);

    run_component_smoke_tests.step.dependOn(&guest_component_smoke.step);
    run_component_smoke_tests.step.dependOn(&gen_component_smoke_wit_run.step);

    const test_component_smoke_step = b.step("test-component-smoke", "Generate WIT, build a smoke guest, and run it through zug's WASM runtime");
    test_component_smoke_step.dependOn(&run_component_smoke_tests.step);

    const run_basic_guest = b.addRunArtifact(exe);
    run_basic_guest.addArgs(&.{ "wasm", "zig-out/basic.wasm", "--arg", "7" });
    run_basic_guest.step.dependOn(&guest_basic.step);

    const run_wasi_log_guest = b.addRunArtifact(exe);
    run_wasi_log_guest.addArgs(&.{ "wasm", "zig-out/wasi_log.wasm" });
    run_wasi_log_guest.step.dependOn(&guest_wasi_log.step);

    const run_wasi_nn_smoke_guest = b.addRunArtifact(exe);
    run_wasi_nn_smoke_guest.addArgs(&.{ "wasm", "zig-out/wasi_nn_smoke.wasm" });
    run_wasi_nn_smoke_guest.step.dependOn(&guest_wasi_nn_smoke.step);

    const run_wasi_nn_full_guest = b.addRunArtifact(exe);
    run_wasi_nn_full_guest.addArgs(&.{ "wasm", "zig-out/wasi_nn_full.wasm", "--manifest", "guests/wasi_nn_full.zugmanifest", "--model", "models/tiny_mnist.onnx" });
    run_wasi_nn_full_guest.step.dependOn(&guest_wasi_nn_full.step);

    const run_gpu_probe_guest = b.addRunArtifact(exe);
    run_gpu_probe_guest.addArgs(&.{ "wasm", "zig-out/gpu_probe.wasm", "--mock-gpu" });
    run_gpu_probe_guest.step.dependOn(&guest_gpu_probe.step);

    const test_external_wasm_step = b.step("test-external-wasm", "Build and run external wasm guests through zug");
    test_external_wasm_step.dependOn(&run_basic_guest.step);
    test_external_wasm_step.dependOn(&run_wasi_log_guest.step);
    test_external_wasm_step.dependOn(&run_wasi_nn_smoke_guest.step);
    test_external_wasm_step.dependOn(&run_wasi_nn_full_guest.step);
    test_external_wasm_step.dependOn(&run_gpu_probe_guest.step);

    const prove_wasm_ml_abi_step = b.step("prove-wasm-ml-abi", "Prove a WASM guest can drive ONNX execution through the stable WASI-NN ABI");
    prove_wasm_ml_abi_step.dependOn(&run_wasi_nn_abi_tests.step);
    prove_wasm_ml_abi_step.dependOn(&run_deployment_tests.step);
    prove_wasm_ml_abi_step.dependOn(&run_wasi_nn_full_guest.step);

    const check_tiny_mnist_workload = b.addRunArtifact(exe);
    check_tiny_mnist_workload.addArgs(&.{ "check", "examples/workloads/tiny-mnist", "--kind", "workload" });
    check_tiny_mnist_workload.step.dependOn(&guest_wasi_nn_full.step);

    const check_workload_step = b.step("check-workload", "Check the example WASM plus ONNX workload package");
    check_workload_step.dependOn(&check_tiny_mnist_workload.step);

    const run_tiny_mnist_workload = b.addRunArtifact(exe);
    run_tiny_mnist_workload.addArgs(&.{ "run", "examples/workloads/tiny-mnist" });
    run_tiny_mnist_workload.step.dependOn(&guest_wasi_nn_full.step);

    const run_workload_step = b.step("run-workload", "Run the example WASM plus ONNX workload package");
    run_workload_step.dependOn(&run_tiny_mnist_workload.step);

    const gen_proto = b.step("gen-proto", "generates zig files from protobuf definitions");

    const protoc_step = protobuf.RunProtocStep.create(protobuf_dep.builder, target, .{
        .destination_directory = b.path("src/proto"),

        .source_files = &.{
            b.path("protocol/onnx/onnx.proto"),
        },

        .include_directories = &.{
            b.path("protocol"),
        },
    });

    gen_proto.dependOn(&protoc_step.step);

    const gen_wit_run = b.addRunArtifact(exe);
    gen_wit_run.addArg("wit");
    gen_wit_run.addArg("edge-inference");
    gen_wit_run.addArg("--out");
    gen_wit_run.addArg("wit/edge-inference.wit");

    const gen_wit = b.step("gen-wit", "generate WIT interface files from Zig descriptors");
    gen_wit.dependOn(&gen_wit_run.step);
}
