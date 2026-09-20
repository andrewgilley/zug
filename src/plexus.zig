//! Local, versioned Plexus worker. It observes execution; it never judges expected results.
const std = @import("std");
const runtime = @import("wasm/runtime.zig");
const Interpreter = runtime.interpreter.Interpreter;
const Instance = runtime.instance.Instance;
const protocol = "plexus-executor/1";
const linked_protocol = "plexus-executor/2";
const component_protocol = "plexus-executor/3";
const component_runtime = @import("wasm/component_runtime.zig");
const max_linked_modules = 8;
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
        return memoryFields(allocator, value.object);
    }
};

/// A plexus-executor/2 memory input names the module whose memory it fills.
const LinkedMemoryInput = struct {
    module: []const u8,
    input: MemoryInput,

    pub fn jsonParse(allocator: std.mem.Allocator, source: anytype, options: std.json.ParseOptions) std.json.ParseError(@TypeOf(source.*))!@This() {
        const value = try std.json.innerParse(std.json.Value, allocator, source, options);
        if (value != .object or value.object.count() != 4) return error.UnexpectedToken;
        const name = value.object.get("module") orelse return error.MissingField;
        if (name != .string) return error.UnexpectedToken;
        return .{ .module = name.string, .input = try memoryFields(allocator, value.object) };
    }
};

fn memoryFields(allocator: std.mem.Allocator, object: std.json.ObjectMap) std.json.ParseError(std.json.Scanner)!MemoryInput {
    const exported = object.get("export") orelse return error.MissingField;
    const offset = object.get("offset") orelse return error.MissingField;
    if (exported != .string) return error.UnexpectedToken;
    const start = switch (offset) {
        .integer => |number| std.math.cast(usize, number) orelse return error.UnexpectedToken,
        .number_string => |number| std.fmt.parseInt(usize, number, 10) catch return error.UnexpectedToken,
        else => return error.UnexpectedToken,
    };
    const utf8 = object.get("utf8");
    const bytes = object.get("bytes");
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
const Case = struct { name: []const u8, arguments: []const i32, memory: ?MemoryInput = null };
const Limits = struct { fuel_per_case: u64, memory_bytes: usize };
const Request = struct { @"export": []const u8, result_count: usize, cases: []const Case, limits: Limits };
const Envelope = struct { schema_version: u32, protocol: []const u8, module_path: []const u8, module_digest: []const u8, request: Request };
const ModuleRef = struct { name: []const u8, module_path: []const u8, module_digest: []const u8 };
const Entry = struct { module: []const u8, @"export": []const u8 };
const LinkedCase = struct { name: []const u8, arguments: []const i32, memory: ?LinkedMemoryInput = null };
const LinkedRequest = struct { entry: Entry, result_count: usize, cases: []const LinkedCase, limits: Limits };
const LinkedEnvelope = struct { schema_version: u32, protocol: []const u8, modules: []const ModuleRef, request: LinkedRequest };
const LoadedModule = struct { name: []const u8, bytes: []const u8 };
const ComponentArgument = struct {
    type: []const u8,
    value: [4]u8,

    pub fn jsonParse(allocator: std.mem.Allocator, source: anytype, options: std.json.ParseOptions) std.json.ParseError(@TypeOf(source.*))!@This() {
        const value = try std.json.innerParse(std.json.Value, allocator, source, options);
        if (value != .object or value.object.count() != 2) return error.UnexpectedToken;
        const type_name = value.object.get("type") orelse return error.MissingField;
        const bytes = value.object.get("value") orelse return error.MissingField;
        if (type_name != .string or bytes != .array or bytes.array.items.len != 4) return error.UnexpectedToken;
        var result: [4]u8 = undefined;
        for (bytes.array.items, &result) |byte, *item| {
            if (byte != .integer) return error.UnexpectedToken;
            item.* = std.math.cast(u8, byte.integer) orelse return error.UnexpectedToken;
        }
        return .{ .type = type_name.string, .value = result };
    }
};
const ComponentCase = struct { name: []const u8, arguments: [1]ComponentArgument };
const ComponentRequest = struct {
    profile: []const u8,
    @"export": []const u8,
    cases: []const ComponentCase,
    limits: Limits,
    host_grants: []const std.json.Value,
};
const ComponentEnvelope = struct { schema_version: u32, protocol: []const u8, component_path: []const u8, component_digest: []const u8, request: ComponentRequest };
const ComponentStage = enum { component_validation, initialization, invocation };
const ComponentOutcome = union(enum) {
    returned: u32,
    fuel_exhausted: ComponentStage,
    failed: struct { stage: ComponentStage, diagnostic: []const u8 },

    pub fn jsonStringify(self: ComponentOutcome, out: *std.json.Stringify) !void {
        switch (self) {
            .returned => |value| try out.write(.{ .kind = "returned", .values = .{.{ .type = "u32", .value = value }} }),
            .fuel_exhausted => |stage| try out.write(.{ .kind = "fuel_exhausted", .stage = stage }),
            .failed => |v| try out.write(.{ .kind = "failed", .stage = v.stage, .diagnostic = v.diagnostic }),
        }
    }
};
const ComponentObservation = struct { name: []const u8, outcome: ComponentOutcome, fuel_consumed: ?u64, host_trace: [0]std.json.Value = .{} };
const ComponentExecution = union(enum) {
    observed: []const ComponentObservation,
    unsupported: []const u8,

    pub fn jsonStringify(self: ComponentExecution, out: *std.json.Stringify) !void {
        switch (self) {
            .observed => |cases| try out.write(.{ .kind = "observed", .cases = cases }),
            .unsupported => |reason| try out.write(.{ .kind = "unsupported", .reason = reason }),
        }
    }
};
const ModuleEcho = struct { name: []const u8, module_digest: []const u8 };
const Stage = enum { link, initialization, input, invocation };
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
            .capabilities = .{ .fuel = true, .memory_limit = true, .max_results = 1, .memory_input = true, .memory_bytes = true, .linked_modules = true, .component_fixed_list = true },
        });
        return;
    }
    if (!std.mem.eql(u8, command, "execute")) return error.ExpectedDescribeOrExecute;
    const path = args.next() orelse return error.MissingRequestPath;
    if (args.next() != null) return error.UnexpectedArgument;
    if (!std.fs.path.isAbsolute(path)) return error.AbsolutePathRequired;
    const json = try std.Io.Dir.cwd().readFileAlloc(std.Options.debug_io, path, allocator, .limited(max_request_bytes));
    if (try requestProtocolIs(allocator, json, component_protocol)) return executeComponentRequest(allocator, json);
    if (try requestProtocolIs(allocator, json, linked_protocol)) return executeLinkedRequest(allocator, json);
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

fn requestProtocolIs(allocator: std.mem.Allocator, json: []const u8, expected: []const u8) !bool {
    const value = try std.json.parseFromSlice(std.json.Value, allocator, json, .{});
    if (value.value != .object) return error.UnexpectedToken;
    const field = value.value.object.get("protocol") orelse return error.MissingField;
    return field == .string and std.mem.eql(u8, field.string, expected);
}

fn executeComponentRequest(allocator: std.mem.Allocator, json: []const u8) !void {
    // Zig's typed JSON parser can coerce numeric strings and integral floats.
    // The wire contract requires actual JSON integer tokens for these fields.
    const raw = try std.json.parseFromSlice(std.json.Value, allocator, json, .{});
    if (raw.value != .object) return error.InvalidComponentEnvelope;
    const version = raw.value.object.get("schema_version") orelse return error.MissingSchemaVersion;
    if (version != .integer or version.integer != 1) return error.UnsupportedProtocol;
    const raw_request = raw.value.object.get("request") orelse return error.InvalidComponentRequest;
    if (raw_request != .object) return error.InvalidComponentRequest;
    const raw_limits = raw_request.object.get("limits") orelse return error.InvalidComponentLimits;
    if (raw_limits != .object) return error.InvalidComponentLimits;
    for ([_][]const u8{ "fuel_per_case", "memory_bytes" }) |field| {
        const number = raw_limits.object.get(field) orelse return error.InvalidComponentLimits;
        if (number != .integer) return error.InvalidComponentLimits;
    }
    const parsed = try std.json.parseFromSlice(ComponentEnvelope, allocator, json, .{});
    const envelope = parsed.value;
    if (envelope.schema_version != 1 or !std.mem.eql(u8, envelope.protocol, component_protocol)) return error.UnsupportedProtocol;
    if (!std.fs.path.isAbsolute(envelope.component_path)) return error.AbsolutePathRequired;
    try validateComponentRequest(envelope.request);
    const bytes = try std.Io.Dir.cwd().readFileAlloc(std.Options.debug_io, envelope.component_path, allocator, .limited(1024 * 1024));
    const digest = moduleDigest(bytes);
    if (!std.mem.eql(u8, &digest, envelope.component_digest)) return error.ComponentDigestMismatch;
    const execution = try executeComponent(allocator, bytes, envelope.request);
    try emit(allocator, .{ .schema_version = 1, .protocol = component_protocol, .component_digest = &digest, .execution = execution });
}

fn validateComponentRequest(request: ComponentRequest) !void {
    if (!std.mem.eql(u8, request.profile, "fixed-list-u8-u32/1")) return error.UnsupportedComponentProfile;
    if (request.host_grants.len != 0) return error.UnsupportedHostGrants;
    if (request.@"export".len == 0 or request.@"export".len > 1024) return error.InvalidExport;
    if (request.cases.len == 0 or request.cases.len > 64) return error.InvalidCaseCount;
    if (request.limits.fuel_per_case == 0 or request.limits.fuel_per_case > 10_000_000) return error.InvalidFuelLimit;
    if (request.limits.memory_bytes < page or request.limits.memory_bytes > 64 * 1024 * 1024) return error.InvalidMemoryLimit;
    for (request.cases, 0..) |case, index| {
        if (case.name.len == 0 or case.name.len > 128) return error.InvalidCase;
        if (!std.mem.eql(u8, case.arguments[0].type, "list<u8,4>")) return error.InvalidComponentArgumentType;
        for (request.cases[0..index]) |prior| {
            if (std.mem.eql(u8, prior.name, case.name)) return error.DuplicateCaseName;
        }
    }
}

fn executeComponent(allocator: std.mem.Allocator, bytes: []const u8, request: ComponentRequest) !ComponentExecution {
    try validateComponentRequest(request);
    const resolved = component_runtime.resolve(allocator, bytes, request.@"export") catch |err| return .{ .unsupported = @errorName(err) };
    const rt = runtime.Runtime.init(allocator);
    var parsed = rt.parseModule(resolved.module_bytes) catch |err| return .{ .unsupported = @errorName(err) };
    defer parsed.deinit(allocator);
    // The supported canonical ABI flattens the value; memory is never used to
    // invent a pointer convention at the component boundary.
    if (parsed.imports.items.len != 0 or parsed.tables.items.len != 0 or parsed.memories.items.len > 1) return .{ .unsupported = "component core imports, tables or multiple memories are outside this profile" };
    const initial_bytes: usize = if (parsed.memories.items.len == 0) 0 else @as(usize, parsed.memories.items[0].limits.min) * page;
    if (initial_bytes > request.limits.memory_bytes) return .{ .unsupported = "minimum memory exceeds requested budget" };
    rt.validateModule(&parsed) catch |err| return .{ .unsupported = @errorName(err) };
    const observations = try allocator.alloc(ComponentObservation, request.cases.len);
    for (request.cases, observations) |case, *observation| {
        observation.* = try observeComponent(&parsed, initial_bytes, resolved.core_export, request.limits, case);
    }
    return .{ .observed = observations };
}

fn observeComponent(parsed: *const runtime.module.Module, initial_bytes: usize, core_export: []const u8, limits: Limits, case: ComponentCase) !ComponentObservation {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const rt = runtime.Runtime.init(arena.allocator());
    var instance = rt.instantiate(parsed, initial_bytes) catch |err| return .{ .name = case.name, .outcome = componentFailure(err, .initialization), .fuel_consumed = 0 };
    defer instance.deinit();
    instance.max_memory_bytes = limits.memory_bytes;
    var interpreter = Interpreter.init(&instance);
    interpreter.fuel_remaining = limits.fuel_per_case;
    interpreter.runStart() catch |err| return .{ .name = case.name, .outcome = componentFailure(err, .initialization), .fuel_consumed = interpreter.fuel_consumed };
    var arguments: [4]runtime.interpreter.Value = undefined;
    for (case.arguments[0].value, &arguments) |byte, *argument| argument.* = .{ .i32 = byte };
    const returned = interpreter.callExport(core_export, &arguments) catch |err| return .{ .name = case.name, .outcome = componentFailure(err, .invocation), .fuel_consumed = interpreter.fuel_consumed };
    const value = returned orelse return error.UnexpectedRuntimeResult;
    if (value != .i32) return error.UnexpectedRuntimeResult;
    // i32 carries the bit pattern of a canonical unsigned u32 result.
    return .{ .name = case.name, .outcome = .{ .returned = value.i32 }, .fuel_consumed = interpreter.fuel_consumed };
}

fn componentFailure(err: anyerror, stage: ComponentStage) ComponentOutcome {
    if (err == error.FuelExhausted) return .{ .fuel_exhausted = stage };
    return .{ .failed = .{ .stage = stage, .diagnostic = @errorName(err) } };
}

fn executeLinkedRequest(allocator: std.mem.Allocator, json: []const u8) !void {
    const parsed = try std.json.parseFromSlice(LinkedEnvelope, allocator, json, .{});
    const envelope = parsed.value;
    if (envelope.schema_version != 1 or !std.mem.eql(u8, envelope.protocol, linked_protocol)) return error.UnsupportedProtocol;
    try validateLinkedRequest(envelope.modules, envelope.request);
    const modules = try allocator.alloc(LoadedModule, envelope.modules.len);
    const echoes = try allocator.alloc(ModuleEcho, envelope.modules.len);
    for (envelope.modules, modules, echoes) |reference, *loaded, *echo| {
        if (!std.fs.path.isAbsolute(reference.module_path)) return error.AbsolutePathRequired;
        const bytes = try std.Io.Dir.cwd().readFileAlloc(std.Options.debug_io, reference.module_path, allocator, .limited(max_module_bytes));
        const digest = moduleDigest(bytes);
        if (!std.mem.eql(u8, &digest, reference.module_digest)) return error.ModuleDigestMismatch;
        loaded.* = .{ .name = reference.name, .bytes = bytes };
        echo.* = .{ .name = reference.name, .module_digest = reference.module_digest };
    }
    const execution = try executeLinked(allocator, modules, envelope.request);
    try emit(allocator, .{ .schema_version = 1, .protocol = linked_protocol, .modules = echoes, .execution = execution });
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

fn validateLinkedRequest(modules: []const ModuleRef, request: LinkedRequest) !void {
    if (modules.len == 0 or modules.len > max_linked_modules) return error.InvalidModuleCount;
    for (modules, 0..) |module, index| {
        if (module.name.len == 0 or module.name.len > 64) return error.InvalidModuleName;
        for (module.name) |byte| {
            if (!std.ascii.isAlphanumeric(byte) and byte != '_' and byte != '-') return error.InvalidModuleName;
        }
        for (modules[0..index]) |prior| {
            if (std.mem.eql(u8, prior.name, module.name)) return error.DuplicateModuleName;
        }
    }
    if (moduleIndex(modules, request.entry.module) == null) return error.UnknownEntryModule;
    if (request.entry.@"export".len == 0 or request.result_count > 1) return error.InvalidExportOrResultCount;
    if (request.cases.len == 0 or request.cases.len > 64) return error.InvalidCaseCount;
    if (request.limits.fuel_per_case == 0 or request.limits.fuel_per_case > 10_000_000) return error.InvalidFuelLimit;
    if (request.limits.memory_bytes < page or request.limits.memory_bytes > 64 * 1024 * 1024) return error.InvalidMemoryLimit;
    for (request.cases, 0..) |case, index| {
        if (case.name.len == 0 or case.arguments.len > 16) return error.InvalidCase;
        for (request.cases[0..index]) |prior| {
            if (std.mem.eql(u8, prior.name, case.name)) return error.DuplicateCaseName;
        }
        if (case.memory) |memory| {
            if (moduleIndex(modules, memory.module) == null) return error.UnknownMemoryModule;
            if (memory.input.@"export".len == 0 or memory.input.payload.len > 1024 * 1024) return error.InvalidMemoryInput;
            const end = std.math.add(usize, memory.input.offset, memory.input.payload.len) catch return error.InvalidMemoryInput;
            if (end > request.limits.memory_bytes) return error.InvalidMemoryInput;
        }
    }
}

fn moduleIndex(modules: anytype, name: []const u8) ?usize {
    for (modules, 0..) |module, index| {
        if (std.mem.eql(u8, module.name, name)) return index;
    }
    return null;
}

/// Instantiate the modules in order for every case, binding each module's
/// function imports to exports of earlier modules. Unsupported shapes are
/// reported as runtime limitations before any guest instruction runs.
fn executeLinked(allocator: std.mem.Allocator, modules: []const LoadedModule, request: LinkedRequest) !Execution {
    const rt = runtime.Runtime.init(allocator);
    const parsed = try allocator.alloc(runtime.module.Module, modules.len);
    const initial = try allocator.alloc(usize, modules.len);
    for (modules, parsed, initial) |module, *out, *bytes| {
        out.* = rt.parseModule(module.bytes) catch |err| return .{ .unsupported = @errorName(err) };
        for (out.imports.items) |import| {
            // A module may import the memory an earlier module exports; tables
            // and globals still cross no module boundary in this worker.
            if (import.kind != .function and import.kind != .memory) return .{ .unsupported = "imported tables and globals are not supported by this worker" };
        }
        if (out.tables.items.len != 0) return .{ .unsupported = "tables are not supported by this worker" };
        if (runtime.instance.memoryCount(out) > 1) return .{ .unsupported = "multiple memories are not supported" };
        bytes.* = if (out.memories.items.len == 0) 0 else @as(usize, out.memories.items[0].limits.min) * page;
        if (bytes.* > request.limits.memory_bytes) return .{ .unsupported = "minimum memory exceeds requested budget" };
        rt.validateModule(out) catch |err| return .{ .unsupported = @errorName(err) };
    }
    const entry = moduleIndex(modules, request.entry.module).?;
    var selected: ?runtime.module.FunctionType = null;
    for (parsed[entry].exports.items) |exported| {
        if (exported.kind == .function and std.mem.eql(u8, exported.name, request.entry.@"export")) {
            const defined = std.math.sub(u32, exported.index, @intCast(importedFunctions(&parsed[entry]))) catch {
                return .{ .unsupported = "re-exported imports are not supported as the entry" };
            };
            selected = try parsed[entry].functionType(parsed[entry].functions.items[defined].type_index);
        }
    }
    const signature = selected orelse return .{ .unsupported = "unknown function export" };
    if (signature.results.len != request.result_count) return error.ResultArityMismatch;
    for (signature.params) |param| if (param != .i32) return .{ .unsupported = "only i32 parameters are supported" };
    for (signature.results) |result| if (result != .i32) return .{ .unsupported = "only i32 results are supported" };
    for (request.cases) |case| if (case.arguments.len != signature.params.len) return error.ArgumentArityMismatch;

    const observations = try allocator.alloc(Observation, request.cases.len);
    for (request.cases, observations) |case, *observation| {
        observation.* = try observeLinked(allocator, modules, parsed, initial, entry, request, case);
    }
    return .{ .observed = observations };
}

fn importedFunctions(module: *const runtime.module.Module) usize {
    var count: usize = 0;
    for (module.imports.items) |import| {
        if (import.kind == .function) count += 1;
    }
    return count;
}

fn observeLinked(
    allocator: std.mem.Allocator,
    modules: []const LoadedModule,
    parsed: []const runtime.module.Module,
    initial: []const usize,
    entry: usize,
    request: LinkedRequest,
    case: LinkedCase,
) !Observation {
    const values = try allocator.alloc(i32, request.result_count);
    var case_arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer case_arena.deinit();
    const scratch = case_arena.allocator();
    const rt = runtime.Runtime.init(scratch);

    // Instances need stable addresses: later instances call into earlier ones.
    const instances = try scratch.alloc(Instance, modules.len);
    const named = try scratch.alloc(runtime.instance.NamedInstance, modules.len);
    var fuel_remaining = request.limits.fuel_per_case;
    var fuel_consumed: u64 = 0;
    for (modules, parsed, initial, instances, named, 0..) |module, *module_parsed, initial_bytes, *instance, *name, index| {
        // Resolving an imported memory is part of linking, so a missing or
        // incompatible memory export is reported before instantiation.
        const imported_memory = runtime.instance.resolveImportedMemory(module_parsed, named[0..index]) catch |err| return .{ .name = case.name, .outcome = failure(err, .link), .fuel_consumed = fuel_consumed };
        instance.* = rt.instantiateWithMemory(module_parsed, initial_bytes, imported_memory) catch |err| return .{ .name = case.name, .outcome = failure(err, .initialization), .fuel_consumed = fuel_consumed };
        instance.max_memory_bytes = request.limits.memory_bytes;
        instance.bindLinkedImports(named[0..index]) catch |err| return .{ .name = case.name, .outcome = failure(err, .link), .fuel_consumed = fuel_consumed };
        name.* = .{ .name = module.name, .instance = instance };

        // Start functions run in order and share the case's fuel budget.
        var starter = Interpreter.init(instance);
        starter.fuel_remaining = fuel_remaining;
        const started = starter.runStart();
        fuel_remaining = starter.fuel_remaining.?;
        fuel_consumed += starter.fuel_consumed;
        started catch |err| return .{ .name = case.name, .outcome = failure(err, .initialization), .fuel_consumed = fuel_consumed };
    }

    if (case.memory) |memory| {
        const target = moduleIndex(modules, memory.module).?;
        var found = false;
        for (parsed[target].exports.items) |exported| {
            if (exported.kind == .memory and exported.index == 0 and std.mem.eql(u8, exported.name, memory.input.@"export")) found = true;
        }
        if (!found) return .{ .name = case.name, .outcome = failure(error.UnknownMemoryExport, .input), .fuel_consumed = fuel_consumed };
        instances[target].memory().write(@intCast(memory.input.offset), memory.input.payload) catch |err| return .{ .name = case.name, .outcome = failure(err, .input), .fuel_consumed = fuel_consumed };
    }

    var interpreter = Interpreter.init(&instances[entry]);
    interpreter.fuel_remaining = fuel_remaining;
    var arguments: [16]runtime.interpreter.Value = undefined;
    for (case.arguments, 0..) |arg, index| arguments[index] = .{ .i32 = @bitCast(arg) };
    const result = interpreter.callExport(request.entry.@"export", arguments[0..case.arguments.len]) catch |err| return .{ .name = case.name, .outcome = failure(err, .invocation), .fuel_consumed = fuel_consumed + interpreter.fuel_consumed };
    fuel_consumed += interpreter.fuel_consumed;
    if (result) |value| {
        if (values.len != 1 or value != .i32) return error.UnexpectedRuntimeResult;
        values[0] = @bitCast(value.i32);
    } else if (values.len != 0) return error.UnexpectedRuntimeResult;
    return .{ .name = case.name, .outcome = .{ .returned = values }, .fuel_consumed = fuel_consumed };
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
        instance.memory().write(@intCast(memory.offset), memory.payload) catch |err| return .{ .name = case.name, .outcome = failure(err, .input), .fuel_consumed = interpreter.fuel_consumed };
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

fn linkedTestRequest(cases: []const LinkedCase, fuel: u64) LinkedRequest {
    return .{ .entry = .{ .module = "consumer", .@"export" = "run" }, .result_count = 1, .cases = cases, .limits = .{ .fuel_per_case = fuel, .memory_bytes = page } };
}

const linked_cases = [_]LinkedCase{
    .{ .name = "all-high", .arguments = &.{0}, .memory = .{ .module = "consumer", .input = .{ .@"export" = "memory", .offset = 0, .payload = &.{ 255, 255, 255, 255 } } } },
    .{ .name = "unaligned", .arguments = &.{1}, .memory = .{ .module = "consumer", .input = .{ .@"export" = "memory", .offset = 1, .payload = &.{ 1, 2, 4, 8 } } } },
};

test "linked consumer calls the separately built provider through its import" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const modules = [_]LoadedModule{
        .{ .name = "provider", .bytes = runtime.fixtures.linked_provider },
        .{ .name = "consumer", .bytes = runtime.fixtures.linked_consumer },
    };
    const result = try executeLinked(arena.allocator(), &modules, linkedTestRequest(&linked_cases, 10_000));
    try std.testing.expectEqual(@as(i32, 1020), result.observed[0].outcome.returned[0]);
    try std.testing.expectEqual(@as(i32, 15), result.observed[1].outcome.returned[0]);
    // The provider's instructions draw on the same per-case budget.
    try std.testing.expect(result.observed[0].fuel_consumed.? > 15);
}

test "same interface with different behavior links and returns its own results" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const modules = [_]LoadedModule{
        .{ .name = "provider", .bytes = runtime.fixtures.linked_provider_xor },
        .{ .name = "consumer", .bytes = runtime.fixtures.linked_consumer },
    };
    const result = try executeLinked(arena.allocator(), &modules, linkedTestRequest(&linked_cases, 10_000));
    try std.testing.expectEqual(@as(i32, 0), result.observed[0].outcome.returned[0]);
    try std.testing.expectEqual(@as(i32, 15), result.observed[1].outcome.returned[0]);
}

test "mistyped and missing imports fail at the link stage" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const mistyped = [_]LoadedModule{
        .{ .name = "provider", .bytes = runtime.fixtures.linked_provider },
        .{ .name = "consumer", .bytes = runtime.fixtures.linked_consumer_pointer },
    };
    const mismatch = try executeLinked(arena.allocator(), &mistyped, linkedTestRequest(&linked_cases, 10_000));
    try std.testing.expectEqual(Stage.link, mismatch.observed[0].outcome.failed.stage);
    try std.testing.expectEqualStrings("IncompatibleImportType", mismatch.observed[0].outcome.failed.diagnostic);

    const alone = [_]LoadedModule{.{ .name = "consumer", .bytes = runtime.fixtures.linked_consumer }};
    const missing = try executeLinked(arena.allocator(), &alone, linkedTestRequest(&linked_cases, 10_000));
    try std.testing.expectEqualStrings("UnresolvedFunctionImport", missing.observed[0].outcome.failed.diagnostic);
}

const shared_memory_cases = [_]LinkedCase{
    .{ .name = "stamped", .arguments = &.{0}, .memory = .{ .module = "provider", .input = .{ .@"export" = "memory", .offset = 0, .payload = &.{ 1, 2, 4, 8 } } } },
    .{ .name = "consumer-data-segment", .arguments = &.{64} },
};

test "a consumer reads and writes the memory its provider exports" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const modules = [_]LoadedModule{
        .{ .name = "provider", .bytes = runtime.fixtures.shared_memory_provider },
        .{ .name = "consumer", .bytes = runtime.fixtures.shared_memory_consumer },
    };
    const result = try executeLinked(arena.allocator(), &modules, linkedTestRequest(&shared_memory_cases, 10_000));
    // The consumer stamps the first byte, so the provider sees 2, 2, 4, 8.
    try std.testing.expectEqual(@as(i32, 16), result.observed[0].outcome.returned[0]);
    // Nothing was written for this case: the bytes are the consumer's own data
    // segment, which only a shared memory could have carried to the provider.
    try std.testing.expectEqual(@as(i32, 11), result.observed[1].outcome.returned[0]);
}

test "an incompatible memory import fails at the link stage" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const modules = [_]LoadedModule{
        .{ .name = "provider", .bytes = runtime.fixtures.shared_memory_provider },
        .{ .name = "consumer", .bytes = runtime.fixtures.shared_memory_consumer_two_pages },
    };
    const result = try executeLinked(arena.allocator(), &modules, linkedTestRequest(&shared_memory_cases, 10_000));
    try std.testing.expectEqual(Stage.link, result.observed[0].outcome.failed.stage);
    try std.testing.expectEqualStrings("IncompatibleMemoryLimits", result.observed[0].outcome.failed.diagnostic);

    const alone = [_]LoadedModule{.{ .name = "consumer", .bytes = runtime.fixtures.shared_memory_consumer }};
    const missing = try executeLinked(arena.allocator(), &alone, linkedTestRequest(&shared_memory_cases, 10_000));
    try std.testing.expectEqualStrings("UnresolvedMemoryImport", missing.observed[0].outcome.failed.diagnostic);
}

test "fuel is shared across linked modules" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const modules = [_]LoadedModule{
        .{ .name = "provider", .bytes = runtime.fixtures.linked_provider },
        .{ .name = "consumer", .bytes = runtime.fixtures.linked_consumer },
    };
    const baseline = try executeLinked(arena.allocator(), &modules, linkedTestRequest(&linked_cases, 10_000));
    const short = try executeLinked(arena.allocator(), &modules, linkedTestRequest(&linked_cases, baseline.observed[0].fuel_consumed.? - 1));
    try std.testing.expectEqual(Stage.invocation, short.observed[0].outcome.fuel_exhausted);
}
