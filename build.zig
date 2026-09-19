const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const plexus_worker = b.addExecutable(.{
        .name = "zug-plexus",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/plexus.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    const plexus_install = b.addInstallArtifact(plexus_worker, .{});
    b.getInstallStep().dependOn(&plexus_install.step);
    b.step("plexus", "Build the local Plexus executor bridge").dependOn(&plexus_install.step);
    const plexus_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/plexus.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    const run_plexus_tests = b.addRunArtifact(plexus_tests);
    b.step("test-plexus", "Test Plexus bridge and runtime").dependOn(&run_plexus_tests.step);

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

    const runtime_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/tests.zig"),
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
    const test_step = b.step("test", "Run runtime, CLI and Plexus bridge tests");
    test_step.dependOn(&b.addRunArtifact(runtime_tests).step);
    test_step.dependOn(&b.addRunArtifact(cli_tests).step);
    test_step.dependOn(&run_plexus_tests.step);
}
