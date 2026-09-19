//! Local, versioned Plexus worker. It observes execution; it never judges expected results.
const std = @import("std");
const runtime = @import("wasm/runtime.zig");
const Interpreter = runtime.interpreter.Interpreter;
const protocol = "plexus-executor/1";
const page = 65536;
const max_module_bytes = 4 * 1024 * 1024;
const max_request_bytes = 8 * 1024 * 1024;

const MemoryInput = struct {
    @"export": []const u8,
    offset: usize,
    payload: []const u8,

    pub fn jsonParse(allocator: std.mem.Allocator, source: anytype, options: std.json.ParseOptions) std.json.ParseError(@TypeOf(source.*))!@This() {
        const value = try std.json.innerParse(std.json.Value, allocator, source, options);
        if (value != .object or value.object.count() != 3) return error.UnexpectedToken;
        const exported = value.object.get("export") orelse return error.MissingField;
        const offset = value.object.get("offset") orelse return error.MissingField;
        if (exported != .string) return error.UnexpectedToken;
        const start = switch (offset) {
            .integer => |number| std.math.cast(usize, number) orelse return error.UnexpectedToken,
            .number_string => |number| std.fmt.parseInt(usize, number, 10) catch return error.UnexpectedToken,
            else => return error.UnexpectedToken,
        };
        const utf8 = value.object.get("utf8");
        const bytes = value.object.get("bytes");
        if ((utf8 == null) == (bytes == null)) return error.UnexpectedToken;
        const payload = if (utf8) |text| blk: {
            if (text != .string) return error.UnexpectedToken;
            break :blk text.string;
        } else blk: {
            const raw = bytes.?;
            if (raw != .array) return error.UnexpectedToken;
            const result = try allocator.alloc(u8, raw.array.items.len);
            for (raw.array.items, result) |byte, *out| {
                if (byte != .integer) return error.UnexpectedToken;
                out.* = std.math.cast(u8, byte.integer) orelse return error.UnexpectedToken;
            }
            break :blk result;
        };
        return .{ .@"export" = exported.string, .offset = start, .payload = payload };
    }
};
const Case = struct { name: []const u8, arguments: []const i32, memory: ?MemoryInput = null };
const Limits = struct { fuel_per_case: u64, memory_bytes: usize };
const Request = struct { @"export": []const u8, result_count: usize, cases: []const Case, limits: Limits };
const Envelope = struct { schema_version: u32, protocol: []const u8, module_path: []const u8, module_digest: []const u8, request: Request };
const Stage = enum { initialization, input, invocation };
const Outcome = union(enum) {
    returned: []const i32,
    fuel_exhausted: Stage,
    trapped: struct { stage: Stage, code: []const u8, diagnostic: []const u8 },
    failed: struct { stage: Stage, diagnostic: []const u8 },

    pub fn jsonStringify(self: Outcome, out: *std.json.Stringify) !void {
        switch (self) {
            .returned => |values| try out.write(.{ .kind = "returned", .values = values }),
            .fuel_exhausted => |stage| try out.write(.{ .kind = "fuel_exhausted", .stage = stage }),
            .trapped => |v| try out.write(.{ .kind = "trapped", .stage = v.stage, .code = v.code, .diagnostic = v.diagnostic }),
            .failed => |v| try out.write(.{ .kind = "failed", .stage = v.stage, .diagnostic = v.diagnostic }),
        }
    }
};
const Observation = struct { name: []const u8, outcome: Outcome, fuel_consumed: ?u64 };
const Execution = union(enum) {
    observed: []const Observation,
    unsupported: []const u8,

    pub fn jsonStringify(self: Execution, out: *std.json.Stringify) !void {
        switch (self) {
            .observed => |cases| try out.write(.{ .kind = "observed", .cases = cases }),
            .unsupported => |reason| try out.write(.{ .kind = "unsupported", .reason = reason }),
        }
    }
};

pub fn main(init: std.process.Init.Minimal) void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    run(init, arena.allocator()) catch |err| {
        std.debug.print("zug-plexus: {s}\n", .{@errorName(err)});
        std.process.exit(1);
    };
}

fn run(init: std.process.Init.Minimal, allocator: std.mem.Allocator) !void {
    var args = try init.args.iterateAllocator(allocator);
    defer args.deinit();
    _ = args.skip();
    const command = args.next() orelse return error.ExpectedDescribeOrExecute;
    if (std.mem.eql(u8, command, "describe")) {
        if (args.next() != null) return error.UnexpectedArgument;
        try emit(allocator, .{
            .schema_version = 1,
            .protocol = protocol,
            .backend = "zug",
            .version = "0.0.1",
            .capabilities = .{ .fuel = true, .memory_limit = true, .max_results = 1, .memory_input = true, .memory_bytes = true },
        });
        return;
    }
    if (!std.mem.eql(u8, command, "execute")) return error.ExpectedDescribeOrExecute;
    const path = args.next() orelse return error.MissingRequestPath;
    if (args.next() != null) return error.UnexpectedArgument;
    if (!std.fs.path.isAbsolute(path)) return error.AbsolutePathRequired;
    const json = try std.Io.Dir.cwd().readFileAlloc(std.Options.debug_io, path, allocator, .limited(max_request_bytes));
    const parsed = try std.json.parseFromSlice(Envelope, allocator, json, .{});
    const envelope = parsed.value;
    if (envelope.schema_version != 1 or !std.mem.eql(u8, envelope.protocol, protocol)) return error.UnsupportedProtocol;
    if (!std.fs.path.isAbsolute(envelope.module_path)) return error.AbsolutePathRequired;
    try validateRequest(envelope.request);
    const bytes = try std.Io.Dir.cwd().readFileAlloc(std.Options.debug_io, envelope.module_path, allocator, .limited(max_module_bytes));
    const digest = moduleDigest(bytes);
    if (!std.mem.eql(u8, &digest, envelope.module_digest)) return error.ModuleDigestMismatch;
    const execution = try execute(allocator, bytes, envelope.request);
    try emit(allocator, .{ .schema_version = 1, .protocol = protocol, .module_digest = envelope.module_digest, .execution = execution });
}

fn emit(allocator: std.mem.Allocator, value: anytype) !void {
    const json = try std.json.Stringify.valueAlloc(allocator, value, .{});
    defer allocator.free(json);
    try std.Io.File.stdout().writeStreamingAll(std.Options.debug_io, json);
    try std.Io.File.stdout().writeStreamingAll(std.Options.debug_io, "\n");
}

fn moduleDigest(bytes: []const u8) [71]u8 {
    var hash: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &hash, .{});
    return "sha256:".* ++ std.fmt.bytesToHex(hash, .lower);
}

fn validateRequest(request: Request) !void {
    if (request.@"export".len == 0 or request.result_count > 1) return error.InvalidExportOrResultCount;
    if (request.cases.len == 0 or request.cases.len > 64) return error.InvalidCaseCount;
    if (request.limits.fuel_per_case == 0 or request.limits.fuel_per_case > 10_000_000) return error.InvalidFuelLimit;
    if (request.limits.memory_bytes < page or request.limits.memory_bytes > 64 * 1024 * 1024) return error.InvalidMemoryLimit;
    for (request.cases, 0..) |case, index| {
        if (case.name.len == 0 or case.arguments.len > 16) return error.InvalidCase;
        for (request.cases[0..index]) |prior| {
            if (std.mem.eql(u8, prior.name, case.name)) return error.DuplicateCaseName;
        }
        if (case.memory) |memory| {
            if (memory.@"export".len == 0 or memory.payload.len > 1024 * 1024) return error.InvalidMemoryInput;
            const end = std.math.add(usize, memory.offset, memory.payload.len) catch return error.InvalidMemoryInput;
            if (end > request.limits.memory_bytes) return error.InvalidMemoryInput;
        }
    }
}

// Parsing and validation deliberately precede every guest instruction. Unsupported
// features remain runtime limitations, not negative evidence for the hypothesis.
fn execute(allocator: std.mem.Allocator, bytes: []const u8, request: Request) !Execution {
    try validateRequest(request);
    const rt = runtime.Runtime.init(allocator);
    var parsed = rt.parseModule(bytes) catch |err| return .{ .unsupported = @errorName(err) };
    defer parsed.deinit(allocator);
    if (parsed.imports.items.len != 0) return .{ .unsupported = "host imports are not supported by this worker" };
    if (parsed.tables.items.len != 0) return .{ .unsupported = "tables are not supported by this worker" };
    if (parsed.memories.items.len > 1) return .{ .unsupported = "multiple memories are not supported" };
    const initial_bytes: usize = if (parsed.memories.items.len == 0) 0 else @as(usize, parsed.memories.items[0].limits.min) * page;
    if (initial_bytes > request.limits.memory_bytes) return .{ .unsupported = "minimum memory exceeds requested budget" };
    rt.validateModule(&parsed) catch |err| return .{ .unsupported = @errorName(err) };
    var selected: ?runtime.module.FunctionType = null;
    for (parsed.exports.items) |exported| {
        if (exported.kind == .function and std.mem.eql(u8, exported.name, request.@"export")) {
            selected = try parsed.functionType(parsed.functions.items[exported.index].type_index);
        }
    }
    const signature = selected orelse return .{ .unsupported = "unknown function export" };
    if (signature.results.len != request.result_count) return error.ResultArityMismatch;
    for (signature.params) |param| if (param != .i32) return .{ .unsupported = "only i32 parameters are supported" };
    for (signature.results) |result| if (result != .i32) return .{ .unsupported = "only i32 results are supported" };
    for (request.cases) |case| if (case.arguments.len != signature.params.len) return error.ArgumentArityMismatch;

    const observations = try allocator.alloc(Observation, request.cases.len);
    for (request.cases, observations) |case, *observation| {
        observation.* = try observe(allocator, &parsed, initial_bytes, request, case);
    }
    return .{ .observed = observations };
}

fn observe(allocator: std.mem.Allocator, parsed: *const runtime.module.Module, initial_bytes: usize, request: Request, case: Case) !Observation {
    // Keep runtime scratch allocations local to one case, including allocations
    // the interpreter cannot reuse, while returned evidence survives the case.
    const values = try allocator.alloc(i32, request.result_count);
    var case_arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer case_arena.deinit();
    const rt = runtime.Runtime.init(case_arena.allocator());
    var instance = rt.instantiate(parsed, initial_bytes) catch |err| return .{ .name = case.name, .outcome = failure(err, .initialization), .fuel_consumed = 0 };
    defer instance.deinit();
    instance.max_memory_bytes = request.limits.memory_bytes;
    var interpreter = Interpreter.init(&instance);
    interpreter.fuel_remaining = request.limits.fuel_per_case;
    interpreter.runStart() catch |err| return .{ .name = case.name, .outcome = failure(err, .initialization), .fuel_consumed = interpreter.fuel_consumed };
    if (case.memory) |memory| {
        var found = false;
        for (parsed.exports.items) |exported| {
            if (exported.kind == .memory and exported.index == 0 and std.mem.eql(u8, exported.name, memory.@"export")) found = true;
        }
        if (!found) return .{ .name = case.name, .outcome = failure(error.UnknownMemoryExport, .input), .fuel_consumed = interpreter.fuel_consumed };
        instance.memory.write(@intCast(memory.offset), memory.payload) catch |err| return .{ .name = case.name, .outcome = failure(err, .input), .fuel_consumed = interpreter.fuel_consumed };
    }
    var arguments: [16]runtime.interpreter.Value = undefined;
    for (case.arguments, 0..) |arg, index| arguments[index] = .{ .i32 = @bitCast(arg) };
    const result = interpreter.callExport(request.@"export", arguments[0..case.arguments.len]) catch |err| return .{ .name = case.name, .outcome = failure(err, .invocation), .fuel_consumed = interpreter.fuel_consumed };
    if (result) |value| {
        if (values.len != 1 or value != .i32) return error.UnexpectedRuntimeResult;
        values[0] = @bitCast(value.i32);
    } else if (values.len != 0) return error.UnexpectedRuntimeResult;
    return .{ .name = case.name, .outcome = .{ .returned = values }, .fuel_consumed = interpreter.fuel_consumed };
}

fn failure(err: anyerror, stage: Stage) Outcome {
    if (err == error.FuelExhausted) return .{ .fuel_exhausted = stage };
    const code: ?[]const u8 = if (stage == .input) null else switch (err) {
        error.UnreachableInstruction => "UnreachableCodeReached",
        error.InvalidMemoryRange => "MemoryOutOfBounds",
        error.IntegerDivideByZero => "IntegerDivisionByZero",
        error.IntegerOverflow => "IntegerOverflow",
        error.InvalidFloatToIntegerConversion => "BadConversionToInteger",
        error.CallStackOverflow => "StackOverflow",
        else => null,
    };
    if (code) |value| return .{ .trapped = .{ .stage = stage, .code = value, .diagnostic = @errorName(err) } };
    return .{ .failed = .{ .stage = stage, .diagnostic = @errorName(err) } };
}

test "bridge returns scalar observations with a fresh instance per case" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const request = testRequest();
    const result = try execute(arena.allocator(), runtime.fixtures.globals_and_start, request);
    try std.testing.expectEqual(@as(i32, 12), result.observed[0].outcome.returned[0]);
    try std.testing.expectEqual(@as(i32, 12), result.observed[1].outcome.returned[0]);
    try std.testing.expect(result.observed[0].fuel_consumed.? > 0);
}

test "fuel covers initialization and invocation" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var request = testRequest();
    const baseline = try execute(arena.allocator(), runtime.fixtures.globals_and_start, request);
    request.limits.fuel_per_case = baseline.observed[0].fuel_consumed.? - 1;
    const shared = try execute(arena.allocator(), runtime.fixtures.globals_and_start, request);
    try std.testing.expectEqual(Stage.invocation, shared.observed[0].outcome.fuel_exhausted);
    request.limits.fuel_per_case = 1;
    const start = try execute(arena.allocator(), runtime.fixtures.globals_and_start, request);
    try std.testing.expectEqual(Stage.initialization, start.observed[0].outcome.fuel_exhausted);
    const call = try execute(arena.allocator(), runtime.fixtures.return_i32_seven, request);
    try std.testing.expectEqual(Stage.invocation, call.observed[0].outcome.fuel_exhausted);
}

test "memory grows only within host budget and starts at declared minimum" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const result = try execute(arena.allocator(), runtime.fixtures.memory_size_and_grow, testRequest());
    // Fixture adds current size to whether growth returned the old size.
    // A one-page host cap denies growth and leaves the result at one.
    try std.testing.expectEqual(@as(i32, 1), result.observed[0].outcome.returned[0]);
}

test "request limits and no-host-import profile are enforced" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var request = testRequest();
    request.limits.fuel_per_case = 0;
    try std.testing.expectError(error.InvalidFuelLimit, execute(arena.allocator(), runtime.fixtures.return_i32_seven, request));
    request = testRequest();
    const result = try execute(arena.allocator(), runtime.fixtures.wasi_fd_write, request);
    try std.testing.expect(result == .unsupported);
}

fn testRequest() Request {
    return .{ .@"export" = "run", .result_count = 1, .cases = &.{ .{ .name = "first", .arguments = &.{} }, .{ .name = "fresh", .arguments = &.{} } }, .limits = .{ .fuel_per_case = 10000, .memory_bytes = page } };
}
