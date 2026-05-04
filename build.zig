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

    const wasi_nn_abi_module = b.createModule(.{
        .root_source_file = b.path("src/wasi_nn_abi.zig"),
        .target = target,
        .optimize = optimize,
    });

    wasi_nn_abi_module.addImport("protobuf", protobuf_dep.module("protobuf"));

    const wasm_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/wasm/runtime.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });

    wasm_tests.root_module.addImport("protobuf", protobuf_dep.module("protobuf"));
    wasm_tests.root_module.addImport("wasi_nn_abi", wasi_nn_abi_module);

    const run_wasi_nn_tests = b.addRunArtifact(wasi_nn_tests);
    const run_wasi_nn_abi_tests = b.addRunArtifact(wasi_nn_abi_tests);
    const run_wasm_tests = b.addRunArtifact(wasm_tests);

    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_wasi_nn_tests.step);
    test_step.dependOn(&run_wasi_nn_abi_tests.step);
    test_step.dependOn(&run_wasm_tests.step);

    const test_wasi_nn_step = b.step("test-wasi-nn", "Run WASI-NN tests");
    test_wasi_nn_step.dependOn(&run_wasi_nn_tests.step);
    test_wasi_nn_step.dependOn(&run_wasi_nn_abi_tests.step);

    const test_wasi_nn_abi_step = b.step("test-wasi-nn-abi", "Run WASI-NN ABI tests");
    test_wasi_nn_abi_step.dependOn(&run_wasi_nn_abi_tests.step);

    const test_wasm_step = b.step("test-wasm", "Run WASM runtime skeleton tests");
    test_wasm_step.dependOn(&run_wasm_tests.step);

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
}
