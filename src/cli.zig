const std = @import("std");
const onnx = @import("proto/onnx.pb.zig");
const tensor = @import("tensor.zig");
const executor = @import("executor.zig");
const capabilities = @import("capabilities.zig");

const max_model_bytes = 100 * 1024 * 1024;

const CliMode = enum {
    run,
    inspect,
};

const CliInput = struct {
    name: []const u8,
    path: []const u8,
};

const CliArgs = struct {
    mode: CliMode = .run,
    model_path: []const u8,
    inputs: std.ArrayList(CliInput) = .empty,

    pub fn deinit(self: *CliArgs, allocator: std.mem.Allocator) void {
        allocator.free(self.model_path);

        for (self.inputs.items) |input| {
            allocator.free(input.name);
            allocator.free(input.path);
        }

        self.inputs.deinit(allocator);
    }
};

pub fn main(init: std.process.Init.Minimal) !void {
    var debug_allocator: std.heap.DebugAllocator(.{}) = .init;
    defer _ = debug_allocator.deinit();

    const allocator = debug_allocator.allocator();

    var cli = try parseArgs(init, allocator);
    defer cli.deinit(allocator);

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

        printUsage();
        return error.UnknownArgument;
    }

    return cli;
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
        \\  zug inspect <model.onnx>
        \\  zug <model.onnx> [--input name=file.f32]
        \\
    , .{});
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
