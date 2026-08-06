const std = @import("std");
const wasm_component = @import("wasm/component.zig");
const wasm_compatibility = @import("wasm/compatibility.zig");
const wasm_imports = @import("wasm/imports.zig");
const wasm_interpreter = @import("wasm/interpreter.zig");
const wasm_manifest = @import("wasm/manifest.zig");
const wasm_runtime = @import("wasm/runtime.zig");
const wit_generator = @import("wit_generator.zig");

const max_wasm_bytes = 100 * 1024 * 1024;
const default_memory_bytes = 16 * 1024 * 1024;
const wasm_page_size = 64 * 1024;

const Command = enum { run, check, wit, help };

const Options = struct {
    command: Command,
    path: ?[]const u8 = null,
    manifest_path: ?[]const u8 = null,
    export_name: ?[]const u8 = null,
    stdin: ?[]const u8 = null,
    output_path: ?[]const u8 = null,
    memory_bytes: ?usize = null,
    args: std.ArrayList(u32) = .empty,

    fn deinit(self: *Options, allocator: std.mem.Allocator) void {
        freeOptional(allocator, self.path);
        freeOptional(allocator, self.manifest_path);
        freeOptional(allocator, self.export_name);
        freeOptional(allocator, self.stdin);
        freeOptional(allocator, self.output_path);
        self.args.deinit(allocator);
    }
};

pub fn main(init: std.process.Init.Minimal) !void {
    var debug_allocator: std.heap.DebugAllocator(.{}) = .init;
    defer _ = debug_allocator.deinit();
    const allocator = debug_allocator.allocator();

    var options = parseArgs(init, allocator) catch |err| {
        printUsage();
        return err;
    };
    defer options.deinit(allocator);

    switch (options.command) {
        .run => try runModule(allocator, options),
        .check => if (!try checkModule(allocator, options)) std.process.exit(1),
        .wit => try wit_generator.run(allocator, .{
            .descriptor = options.path.?,
            .out_path = options.output_path,
        }),
        .help => printUsage(),
    }
}

fn parseArgs(init: std.process.Init.Minimal, allocator: std.mem.Allocator) !Options {
    var args = try init.args.iterateAllocator(allocator);
    defer args.deinit();
    _ = args.skip();

    const command_text = args.next() orelse return .{ .command = .help };
    if (std.mem.eql(u8, command_text, "help") or std.mem.eql(u8, command_text, "--help") or std.mem.eql(u8, command_text, "-h")) {
        return .{ .command = .help };
    }

    const command: Command = if (std.mem.eql(u8, command_text, "run") or std.mem.eql(u8, command_text, "wasm"))
        .run
    else if (std.mem.eql(u8, command_text, "check"))
        .check
    else if (std.mem.eql(u8, command_text, "wit"))
        .wit
    else
        return error.UnknownCommand;

    const path = args.next() orelse return error.MissingPath;
    var options = Options{ .command = command, .path = try allocator.dupe(u8, path) };
    errdefer options.deinit(allocator);

    while (args.next()) |arg| {
        if (std.mem.eql(u8, arg, "--manifest")) {
            try replaceString(allocator, &options.manifest_path, args.next() orelse return error.MissingManifestPath);
        } else if (std.mem.eql(u8, arg, "--export")) {
            try replaceString(allocator, &options.export_name, args.next() orelse return error.MissingExportName);
        } else if (std.mem.eql(u8, arg, "--memory")) {
            options.memory_bytes = try parseByteCount(args.next() orelse return error.MissingMemorySize);
        } else if (std.mem.eql(u8, arg, "--arg")) {
            const raw = args.next() orelse return error.MissingArgumentValue;
            const value = try std.fmt.parseInt(i32, raw, 10);
            try options.args.append(allocator, @bitCast(value));
        } else if (std.mem.eql(u8, arg, "--stdin")) {
            try replaceString(allocator, &options.stdin, args.next() orelse return error.MissingStdinValue);
        } else if (std.mem.eql(u8, arg, "--out") and command == .wit) {
            try replaceString(allocator, &options.output_path, args.next() orelse return error.MissingOutputPath);
        } else {
            return error.UnknownArgument;
        }
    }

    return options;
}

fn runModule(allocator: std.mem.Allocator, options: Options) !void {
    const bytes = try readFile(allocator, options.path.?, max_wasm_bytes);
    defer allocator.free(bytes);
    if (wasm_component.isComponent(bytes)) return error.ComponentExecutionUnsupported;

    const runtime = wasm_runtime.Runtime.init(allocator);
    var parsed = try runtime.parseModule(bytes);
    defer parsed.deinit(allocator);

    var manifest = try loadManifest(allocator, options.manifest_path);
    defer if (manifest) |*loaded| loaded.deinit(allocator);

    const export_name = options.export_name orelse if (manifest) |loaded| loaded.export_name orelse "run" else "run";
    const initial_memory_bytes = try initialMemoryBytes(options.memory_bytes, manifest);

    var report = try wasm_compatibility.analyzeModule(allocator, bytes, &parsed, .{}, .{
        .initial_memory_bytes = initial_memory_bytes,
        .export_name = export_name,
    });
    defer report.deinit(allocator);
    if (!report.supported()) {
        report.print();
        return error.WasmCompatibilityCheckFailed;
    }

    if (manifest) |loaded| try loaded.validate(&parsed, .{}, .{
        .export_name = export_name,
        .initial_memory_bytes = initial_memory_bytes,
    });

    var instance = try runtime.instantiate(&parsed, initial_memory_bytes);
    defer instance.deinit();

    const guest_args = [_][]const u8{"zug"};
    var resolver = wasm_imports.Resolver.initWasiConfig(allocator, &instance.memory, .{
        .args = &guest_args,
        .stdin = options.stdin orelse "",
    });
    defer resolver.deinit();
    try instance.bindImports(&resolver);

    var interpreter = wasm_interpreter.Interpreter.init(&instance);
    try interpreter.runStart();

    var values: std.ArrayList(wasm_interpreter.Value) = .empty;
    defer values.deinit(allocator);
    for (options.args.items) |arg| try values.append(allocator, .{ .i32 = arg });
    const result = try interpreter.callExport(export_name, values.items);

    if (resolver.stdout.items.len != 0) std.debug.print("{s}", .{resolver.stdout.items});
    if (resolver.stderr.items.len != 0) std.debug.print("{s}", .{resolver.stderr.items});
    if (resolver.exit_code) |exit_code| std.debug.print("wasm exit: {d}\n", .{exit_code});
    printValue(result);
}

fn checkModule(allocator: std.mem.Allocator, options: Options) !bool {
    const bytes = try readFile(allocator, options.path.?, max_wasm_bytes);
    defer allocator.free(bytes);

    if (wasm_component.isComponent(bytes)) {
        var component = try wasm_component.Component.parse(allocator, bytes);
        defer component.deinit(allocator);
        std.debug.print("zug check component\n  parse: pass\n", .{});
        return true;
    }

    const runtime = wasm_runtime.Runtime.init(allocator);
    var parsed = try runtime.parseModule(bytes);
    defer parsed.deinit(allocator);
    const initial_memory_bytes = try initialMemoryBytes(options.memory_bytes, null);
    var report = try wasm_compatibility.analyzeModule(allocator, bytes, &parsed, .{}, .{
        .initial_memory_bytes = initial_memory_bytes,
        .export_name = options.export_name,
    });
    defer report.deinit(allocator);
    report.print();
    return report.supported();
}

fn loadManifest(allocator: std.mem.Allocator, path: ?[]const u8) !?wasm_manifest.Manifest {
    const actual = path orelse return null;
    const bytes = try readFile(allocator, actual, wasm_manifest.max_manifest_bytes);
    defer allocator.free(bytes);
    return try wasm_manifest.Manifest.parse(allocator, bytes);
}

fn initialMemoryBytes(requested: ?usize, manifest: ?wasm_manifest.Manifest) !usize {
    var needed = requested orelse default_memory_bytes;
    if (manifest) |loaded| {
        if (loaded.min_memory_bytes) |minimum| needed = @max(needed, minimum);
    }
    const aligned = try alignToWasmPage(needed);
    if (manifest) |loaded| {
        if (loaded.max_memory_bytes) |maximum| {
            if (aligned > maximum) return error.ManifestMemoryTooLarge;
        }
    }
    return aligned;
}

fn alignToWasmPage(value: usize) !usize {
    const plus_page = try std.math.add(usize, value, wasm_page_size - 1);
    return plus_page - (plus_page % wasm_page_size);
}

fn readFile(allocator: std.mem.Allocator, path: []const u8, limit: usize) ![]u8 {
    return std.Io.Dir.cwd().readFileAlloc(std.Options.debug_io, path, allocator, .limited(limit));
}

fn parseByteCount(value: []const u8) !usize {
    const suffixes = [_]struct { suffix: []const u8, multiplier: usize }{
        .{ .suffix = "KiB", .multiplier = 1024 },
        .{ .suffix = "MiB", .multiplier = 1024 * 1024 },
        .{ .suffix = "GiB", .multiplier = 1024 * 1024 * 1024 },
    };
    for (suffixes) |suffix| if (std.mem.endsWith(u8, value, suffix.suffix)) {
        const raw = value[0 .. value.len - suffix.suffix.len];
        return std.math.mul(usize, try std.fmt.parseInt(usize, raw, 10), suffix.multiplier);
    };
    return std.fmt.parseInt(usize, value, 10);
}

fn printValue(result: ?wasm_interpreter.Value) void {
    const value = result orelse {
        std.debug.print("wasm result: <none>\n", .{});
        return;
    };
    std.debug.print("wasm result: ", .{});
    switch (value) {
        .i32 => |actual| std.debug.print("i32 {d}\n", .{@as(i32, @bitCast(actual))}),
        .i64 => |actual| std.debug.print("i64 {d}\n", .{@as(i64, @bitCast(actual))}),
        .f32 => |actual| std.debug.print("f32 {d}\n", .{actual}),
        .f64 => |actual| std.debug.print("f64 {d}\n", .{actual}),
        .funcref => |actual| std.debug.print("funcref {?}\n", .{actual}),
        .v128 => |actual| std.debug.print("v128 {any}\n", .{actual}),
    }
}

fn replaceString(allocator: std.mem.Allocator, target: *?[]const u8, value: []const u8) !void {
    freeOptional(allocator, target.*);
    target.* = try allocator.dupe(u8, value);
}

fn freeOptional(allocator: std.mem.Allocator, value: ?[]const u8) void {
    if (value) |actual| allocator.free(actual);
}

fn printUsage() void {
    std.debug.print(
        \\usage:
        \\  zug run <module.wasm> [--manifest file] [--export name] [--memory bytes] [--arg i32] [--stdin text]
        \\  zug check <module.wasm|component.wasm> [--export name] [--memory bytes]
        \\  zug wit component-smoke [--out file.wit]
        \\
    , .{});
}

test "byte counts align to WebAssembly pages" {
    try std.testing.expectEqual(@as(usize, 2 * 1024 * 1024), try parseByteCount("2MiB"));
    try std.testing.expectEqual(@as(usize, 64 * 1024), try alignToWasmPage(1));
}
