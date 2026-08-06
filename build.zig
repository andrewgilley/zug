const std = @import("std");

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
    b.installArtifact(exe);

    const run_exe = b.addRunArtifact(exe);
    if (b.args) |args| run_exe.addArgs(args);
    const run_step = b.step("run", "Run zug");
    run_step.dependOn(&run_exe.step);

    const wasm_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/wasm_tests.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    const wit_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/wit_generator.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    const cli_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/cli.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    const run_wasm_tests = b.addRunArtifact(wasm_tests);
    const run_wit_tests = b.addRunArtifact(wit_tests);
    const run_cli_tests = b.addRunArtifact(cli_tests);
    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_wasm_tests.step);
    test_step.dependOn(&run_wit_tests.step);
    test_step.dependOn(&run_cli_tests.step);

    const guest_basic = addGuest(b, "guests/basic.zig", "zig-out/basic.wasm");

    const guests_step = b.step("guests", "Build all guest fixtures");
    guests_step.dependOn(&guest_basic.step);

    const run_basic = b.addRunArtifact(exe);
    run_basic.addArgs(&.{ "run", "zig-out/basic.wasm", "--arg", "7" });
    run_basic.step.dependOn(&guest_basic.step);

    const test_guests = b.step("test-guests", "Build and run guest fixtures");
    test_guests.dependOn(&run_basic.step);

    const gen_wit_run = b.addRunArtifact(exe);
    gen_wit_run.addArgs(&.{ "wit", "component-smoke", "--out", "zig-out/component-smoke.wit" });
    const gen_wit = b.step("gen-wit", "Generate the component smoke WIT interface");
    gen_wit.dependOn(&gen_wit_run.step);
}

fn addGuest(b: *std.Build, source: []const u8, output: []const u8) *std.Build.Step.Run {
    return b.addSystemCommand(&.{
        b.graph.zig_exe,
        "build-exe",
        source,
        "-target",
        "wasm32-freestanding",
        "-O",
        "ReleaseSmall",
        "-fno-entry",
        "-rdynamic",
        b.fmt("-femit-bin={s}", .{output}),
    });
}
