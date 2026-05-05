const std = @import("std");
const binary = @import("binary.zig");
const fixtures = @import("fixtures.zig");
const imports = @import("imports.zig");
const instance = @import("instance.zig");
const module = @import("module.zig");

pub const Value = union(enum) {
    i32: u32,
    i64: u64,
    f32: f32,
    f64: f64,
};

const Flow = union(enum) {
    done,
    branch: u32,
    returned: ?Value,
};

const BlockBoundary = struct {
    body_start: usize,
    else_offset: ?usize = null,
    end_offset: usize,
};

pub const Interpreter = struct {
    instance: *instance.Instance,

    pub fn init(wasm_instance: *instance.Instance) Interpreter {
        return .{
            .instance = wasm_instance,
        };
    }

    pub fn callExport(
        self: *Interpreter,
        name: []const u8,
        args: []const Value,
    ) !?Value {
        const function_index = try self.instance.exportedFunctionIndex(name);
        return self.executeFunction(function_index, args, 0);
    }

    pub fn callImport(
        self: *Interpreter,
        function: imports.Function,
        args: []const u32,
    ) !u32 {
        const resolver = self.instance.import_resolver orelse return error.MissingImportResolver;
        return resolver.call(function, args);
    }

    pub fn runStart(self: *Interpreter) !void {
        if (self.instance.start_executed) return;

        const function_index = self.instance.startFunctionIndex() orelse return;
        const result = try self.executeFunction(function_index, &.{}, 0);
        if (result != null) return error.StartFunctionReturnedValue;

        self.instance.start_executed = true;
    }

    fn executeFunction(
        self: *Interpreter,
        function_index: u32,
        args: []const Value,
        depth: usize,
    ) !?Value {
        if (depth > 128) return error.CallStackOverflow;

        const imported_count = self.instance.importedFunctionCount();
        const actual = std.math.cast(usize, function_index) orelse return error.InvalidFunctionIndex;

        if (actual < imported_count) {
            const imported = try self.instance.importedFunction(function_index);
            const function_type = try self.instance.importedFunctionType(function_index);
            try validateFunctionArgs(function_type, args);
            const result = try self.callImportWithValues(imported, args);
            return resultFromImport(function_type, result);
        }

        const function = try self.instance.definedFunction(function_index);
        return self.executeDefinedFunction(function, args, depth);
    }

    fn executeDefinedFunction(
        self: *Interpreter,
        function: module.Function,
        args: []const Value,
        depth: usize,
    ) !?Value {
        const function_type = try self.instance.functionType(function.type_index);
        try validateFunctionArgs(function_type, args);

        var reader = binary.Reader.init(function.body);
        var locals: std.ArrayList(Value) = .empty;
        defer locals.deinit(self.instance.allocator);

        for (args) |arg| {
            try locals.append(self.instance.allocator, arg);
        }

        try readLocalDeclarations(self.instance.allocator, &reader, &locals);

        var stack: std.ArrayList(Value) = .empty;
        defer stack.deinit(self.instance.allocator);

        const flow = try self.executeInstructionRange(&reader, function.body.len, &locals, &stack, depth);

        const result = switch (flow) {
            .done => popOptional(&stack),
            .returned => |value| value,
            .branch => return error.InvalidBranchDepth,
        };

        try validateFunctionResult(function_type, result);
        return result;
    }

    fn executeInstructionRange(
        self: *Interpreter,
        reader: *binary.Reader,
        limit: usize,
        locals: *std.ArrayList(Value),
        stack: *std.ArrayList(Value),
        depth: usize,
    ) anyerror!Flow {
        while (reader.offset < limit) {
            const opcode = try reader.readByte();

            switch (opcode) {
                0x00 => return error.UnreachableInstruction,
                0x01 => {},
                0x02 => {
                    const flow = try self.executeBlock(reader, locals, stack, depth);
                    switch (flow) {
                        .done => {},
                        else => return flow,
                    }
                },
                0x03 => {
                    const flow = try self.executeLoop(reader, locals, stack, depth);
                    switch (flow) {
                        .done => {},
                        else => return flow,
                    }
                },
                0x04 => {
                    const flow = try self.executeIf(reader, locals, stack, depth);
                    switch (flow) {
                        .done => {},
                        else => return flow,
                    }
                },
                0x05 => return error.UnexpectedElse,
                0x0b => return .done,
                0x0c => {
                    const label_index = try reader.readVarU32();
                    return .{ .branch = label_index };
                },
                0x0d => {
                    const label_index = try reader.readVarU32();
                    const condition = try valueAsI32(try popValue(stack));
                    if (condition != 0) return .{ .branch = label_index };
                },
                0x0e => {
                    const label_index = try readBranchTableTarget(reader, stack);
                    return .{ .branch = label_index };
                },
                0x0f => return .{ .returned = popOptional(stack) },
                0x10 => {
                    const target_index = try reader.readVarU32();
                    try self.callFunctionFromStack(target_index, stack, depth + 1);
                },
                0x1a => _ = try popValue(stack),
                0x1b => try pushSelectedValue(self.instance.allocator, stack),
                0x1c => {
                    try readSelectTypeVector(reader);
                    try pushSelectedValue(self.instance.allocator, stack);
                },
                0x20 => {
                    const local_index = try reader.readVarU32();
                    const local = try localValue(locals.items, local_index);
                    try stack.append(self.instance.allocator, local);
                },
                0x21 => {
                    const local_index = try reader.readVarU32();
                    const value = try popValue(stack);
                    try setLocalValue(locals.items, local_index, value);
                },
                0x22 => {
                    const local_index = try reader.readVarU32();
                    const value = try peekValue(stack);
                    try setLocalValue(locals.items, local_index, value);
                },
                0x23 => {
                    const global_index = try reader.readVarU32();
                    const runtime_global = try self.instance.global(global_index);
                    try stack.append(self.instance.allocator, valueFromConst(runtime_global.value));
                },
                0x24 => {
                    const global_index = try reader.readVarU32();
                    const runtime_global = try self.instance.global(global_index);
                    const value = try popValue(stack);
                    try self.instance.setGlobalValue(
                        global_index,
                        try constValueFromValue(value, runtime_global.global_type.value_type),
                    );
                },
                0x28 => {
                    _ = try reader.readVarU32();
                    const offset = try reader.readVarU32();
                    const address = try valueAsI32(try popValue(stack));
                    const effective_address = try std.math.add(u32, address, offset);
                    const value = try self.instance.memory.readU32(effective_address);
                    try stack.append(self.instance.allocator, .{ .i32 = value });
                },
                0x29 => {
                    _ = try reader.readVarU32();
                    const offset = try reader.readVarU32();
                    const address = try valueAsI32(try popValue(stack));
                    const effective_address = try std.math.add(u32, address, offset);
                    const value = readU64Little(try self.instance.memory.read(effective_address, 8));
                    try stack.append(self.instance.allocator, .{ .i64 = value });
                },
                0x2a => {
                    _ = try reader.readVarU32();
                    const offset = try reader.readVarU32();
                    const address = try valueAsI32(try popValue(stack));
                    const effective_address = try std.math.add(u32, address, offset);
                    const bits = try self.instance.memory.readU32(effective_address);
                    try stack.append(self.instance.allocator, .{ .f32 = @bitCast(bits) });
                },
                0x2b => {
                    _ = try reader.readVarU32();
                    const offset = try reader.readVarU32();
                    const address = try valueAsI32(try popValue(stack));
                    const effective_address = try std.math.add(u32, address, offset);
                    const bits = readU64Little(try self.instance.memory.read(effective_address, 8));
                    try stack.append(self.instance.allocator, .{ .f64 = @bitCast(bits) });
                },
                0x2c => {
                    _ = try reader.readVarU32();
                    const offset = try reader.readVarU32();
                    const address = try valueAsI32(try popValue(stack));
                    const effective_address = try std.math.add(u32, address, offset);
                    const bytes = try self.instance.memory.read(effective_address, 1);
                    const signed: i8 = @bitCast(bytes[0]);
                    const widened: i32 = signed;
                    try stack.append(self.instance.allocator, .{ .i32 = @bitCast(widened) });
                },
                0x2d => {
                    _ = try reader.readVarU32();
                    const offset = try reader.readVarU32();
                    const address = try valueAsI32(try popValue(stack));
                    const effective_address = try std.math.add(u32, address, offset);
                    const bytes = try self.instance.memory.read(effective_address, 1);
                    try stack.append(self.instance.allocator, .{ .i32 = bytes[0] });
                },
                0x2e => {
                    _ = try reader.readVarU32();
                    const offset = try reader.readVarU32();
                    const address = try valueAsI32(try popValue(stack));
                    const effective_address = try std.math.add(u32, address, offset);
                    const signed: i16 = @bitCast(readU16Little(try self.instance.memory.read(effective_address, 2)));
                    const widened: i32 = signed;
                    try stack.append(self.instance.allocator, .{ .i32 = @bitCast(widened) });
                },
                0x2f => {
                    _ = try reader.readVarU32();
                    const offset = try reader.readVarU32();
                    const address = try valueAsI32(try popValue(stack));
                    const effective_address = try std.math.add(u32, address, offset);
                    try stack.append(self.instance.allocator, .{
                        .i32 = @as(u32, readU16Little(try self.instance.memory.read(effective_address, 2))),
                    });
                },
                0x30 => {
                    _ = try reader.readVarU32();
                    const offset = try reader.readVarU32();
                    const address = try valueAsI32(try popValue(stack));
                    const effective_address = try std.math.add(u32, address, offset);
                    const bytes = try self.instance.memory.read(effective_address, 1);
                    const signed: i8 = @bitCast(bytes[0]);
                    const widened: i64 = signed;
                    try stack.append(self.instance.allocator, .{ .i64 = @bitCast(widened) });
                },
                0x31 => {
                    _ = try reader.readVarU32();
                    const offset = try reader.readVarU32();
                    const address = try valueAsI32(try popValue(stack));
                    const effective_address = try std.math.add(u32, address, offset);
                    const bytes = try self.instance.memory.read(effective_address, 1);
                    try stack.append(self.instance.allocator, .{ .i64 = bytes[0] });
                },
                0x32 => {
                    _ = try reader.readVarU32();
                    const offset = try reader.readVarU32();
                    const address = try valueAsI32(try popValue(stack));
                    const effective_address = try std.math.add(u32, address, offset);
                    const signed: i16 = @bitCast(readU16Little(try self.instance.memory.read(effective_address, 2)));
                    const widened: i64 = signed;
                    try stack.append(self.instance.allocator, .{ .i64 = @bitCast(widened) });
                },
                0x33 => {
                    _ = try reader.readVarU32();
                    const offset = try reader.readVarU32();
                    const address = try valueAsI32(try popValue(stack));
                    const effective_address = try std.math.add(u32, address, offset);
                    try stack.append(self.instance.allocator, .{
                        .i64 = @as(u64, readU16Little(try self.instance.memory.read(effective_address, 2))),
                    });
                },
                0x34 => {
                    _ = try reader.readVarU32();
                    const offset = try reader.readVarU32();
                    const address = try valueAsI32(try popValue(stack));
                    const effective_address = try std.math.add(u32, address, offset);
                    const signed: i32 = @bitCast(try self.instance.memory.readU32(effective_address));
                    const widened: i64 = signed;
                    try stack.append(self.instance.allocator, .{ .i64 = @bitCast(widened) });
                },
                0x35 => {
                    _ = try reader.readVarU32();
                    const offset = try reader.readVarU32();
                    const address = try valueAsI32(try popValue(stack));
                    const effective_address = try std.math.add(u32, address, offset);
                    try stack.append(self.instance.allocator, .{
                        .i64 = @as(u64, try self.instance.memory.readU32(effective_address)),
                    });
                },
                0x36 => {
                    _ = try reader.readVarU32();
                    const offset = try reader.readVarU32();
                    const value = try valueAsI32(try popValue(stack));
                    const address = try valueAsI32(try popValue(stack));
                    const effective_address = try std.math.add(u32, address, offset);
                    try self.instance.memory.writeU32(effective_address, value);
                },
                0x37 => {
                    _ = try reader.readVarU32();
                    const offset = try reader.readVarU32();
                    const value = try valueAsI64(try popValue(stack));
                    const address = try valueAsI32(try popValue(stack));
                    const effective_address = try std.math.add(u32, address, offset);
                    var bytes: [8]u8 = undefined;
                    writeU64Little(&bytes, value);
                    try self.instance.memory.write(effective_address, &bytes);
                },
                0x38 => {
                    _ = try reader.readVarU32();
                    const offset = try reader.readVarU32();
                    const value = try valueAsF32(try popValue(stack));
                    const address = try valueAsI32(try popValue(stack));
                    const effective_address = try std.math.add(u32, address, offset);
                    try self.instance.memory.writeU32(effective_address, @bitCast(value));
                },
                0x39 => {
                    _ = try reader.readVarU32();
                    const offset = try reader.readVarU32();
                    const value = try valueAsF64(try popValue(stack));
                    const address = try valueAsI32(try popValue(stack));
                    const effective_address = try std.math.add(u32, address, offset);
                    var bytes: [8]u8 = undefined;
                    writeU64Little(&bytes, @bitCast(value));
                    try self.instance.memory.write(effective_address, &bytes);
                },
                0x3a => {
                    _ = try reader.readVarU32();
                    const offset = try reader.readVarU32();
                    const value = try valueAsI32(try popValue(stack));
                    const address = try valueAsI32(try popValue(stack));
                    const effective_address = try std.math.add(u32, address, offset);
                    const bytes = [_]u8{std.math.cast(u8, value & 0xff) orelse unreachable};
                    try self.instance.memory.write(effective_address, &bytes);
                },
                0x3b => {
                    _ = try reader.readVarU32();
                    const offset = try reader.readVarU32();
                    const value = try valueAsI32(try popValue(stack));
                    const address = try valueAsI32(try popValue(stack));
                    const effective_address = try std.math.add(u32, address, offset);
                    var bytes: [2]u8 = undefined;
                    writeU16Little(&bytes, @intCast(value & 0xffff));
                    try self.instance.memory.write(effective_address, &bytes);
                },
                0x3c => {
                    _ = try reader.readVarU32();
                    const offset = try reader.readVarU32();
                    const value = try valueAsI64(try popValue(stack));
                    const address = try valueAsI32(try popValue(stack));
                    const effective_address = try std.math.add(u32, address, offset);
                    const bytes = [_]u8{std.math.cast(u8, value & 0xff) orelse unreachable};
                    try self.instance.memory.write(effective_address, &bytes);
                },
                0x3d => {
                    _ = try reader.readVarU32();
                    const offset = try reader.readVarU32();
                    const value = try valueAsI64(try popValue(stack));
                    const address = try valueAsI32(try popValue(stack));
                    const effective_address = try std.math.add(u32, address, offset);
                    var bytes: [2]u8 = undefined;
                    writeU16Little(&bytes, @intCast(value & 0xffff));
                    try self.instance.memory.write(effective_address, &bytes);
                },
                0x3e => {
                    _ = try reader.readVarU32();
                    const offset = try reader.readVarU32();
                    const value = try valueAsI64(try popValue(stack));
                    const address = try valueAsI32(try popValue(stack));
                    const effective_address = try std.math.add(u32, address, offset);
                    try self.instance.memory.writeU32(effective_address, @intCast(value & 0xffffffff));
                },
                0x3f => {
                    const memory_index = try reader.readByte();
                    if (memory_index != 0) return error.UnsupportedMemoryIndex;
                    try stack.append(self.instance.allocator, .{ .i32 = try self.instance.currentMemoryPages() });
                },
                0x40 => {
                    const memory_index = try reader.readByte();
                    if (memory_index != 0) return error.UnsupportedMemoryIndex;
                    const delta_pages = try valueAsI32(try popValue(stack));
                    const old_pages = try self.instance.growMemory(delta_pages);
                    try stack.append(self.instance.allocator, .{ .i32 = old_pages orelse 0xffffffff });
                },
                0x41 => {
                    const value = try reader.readVarI32();
                    try stack.append(self.instance.allocator, .{ .i32 = @bitCast(value) });
                },
                0x42 => {
                    const value = try reader.readVarI64();
                    try stack.append(self.instance.allocator, .{ .i64 = @bitCast(value) });
                },
                0x43 => {
                    const bits = readU32Little(try reader.readBytes(4));
                    try stack.append(self.instance.allocator, .{ .f32 = @bitCast(bits) });
                },
                0x44 => {
                    const bits = readU64Little(try reader.readBytes(8));
                    try stack.append(self.instance.allocator, .{ .f64 = @bitCast(bits) });
                },
                0x45 => {
                    const value = try valueAsI32(try popValue(stack));
                    try stack.append(self.instance.allocator, .{ .i32 = if (value == 0) 1 else 0 });
                },
                0x46 => try pushI32Comparison(self.instance.allocator, stack, eqU32),
                0x47 => try pushI32Comparison(self.instance.allocator, stack, neU32),
                0x48 => try pushI32SignedComparison(self.instance.allocator, stack, ltI32),
                0x49 => try pushI32Comparison(self.instance.allocator, stack, ltU32),
                0x4a => try pushI32SignedComparison(self.instance.allocator, stack, gtI32),
                0x4b => try pushI32Comparison(self.instance.allocator, stack, gtU32),
                0x4c => try pushI32SignedComparison(self.instance.allocator, stack, leI32),
                0x4d => try pushI32Comparison(self.instance.allocator, stack, leU32),
                0x4e => try pushI32SignedComparison(self.instance.allocator, stack, geI32),
                0x4f => try pushI32Comparison(self.instance.allocator, stack, geU32),
                0x51 => {
                    const rhs = try valueAsI64(try popValue(stack));
                    const lhs = try valueAsI64(try popValue(stack));
                    try stack.append(self.instance.allocator, .{ .i32 = if (lhs == rhs) 1 else 0 });
                },
                0x50 => {
                    const value = try valueAsI64(try popValue(stack));
                    try stack.append(self.instance.allocator, .{ .i32 = if (value == 0) 1 else 0 });
                },
                0x52 => try pushI64Comparison(self.instance.allocator, stack, neU64),
                0x53 => try pushI64SignedComparison(self.instance.allocator, stack, ltI64),
                0x54 => try pushI64Comparison(self.instance.allocator, stack, ltU64),
                0x55 => try pushI64SignedComparison(self.instance.allocator, stack, gtI64),
                0x56 => try pushI64Comparison(self.instance.allocator, stack, gtU64),
                0x57 => try pushI64SignedComparison(self.instance.allocator, stack, leI64),
                0x58 => try pushI64Comparison(self.instance.allocator, stack, leU64),
                0x59 => try pushI64SignedComparison(self.instance.allocator, stack, geI64),
                0x5a => try pushI64Comparison(self.instance.allocator, stack, geU64),
                0x5b => try pushF32Comparison(self.instance.allocator, stack, eqF32),
                0x5c => try pushF32Comparison(self.instance.allocator, stack, neF32),
                0x5d => try pushF32Comparison(self.instance.allocator, stack, ltF32),
                0x5e => try pushF32Comparison(self.instance.allocator, stack, gtF32),
                0x5f => try pushF32Comparison(self.instance.allocator, stack, leF32),
                0x60 => try pushF32Comparison(self.instance.allocator, stack, geF32),
                0x61 => try pushF64Comparison(self.instance.allocator, stack, eqF64),
                0x62 => try pushF64Comparison(self.instance.allocator, stack, neF64),
                0x63 => try pushF64Comparison(self.instance.allocator, stack, ltF64),
                0x64 => try pushF64Comparison(self.instance.allocator, stack, gtF64),
                0x65 => try pushF64Comparison(self.instance.allocator, stack, leF64),
                0x66 => try pushF64Comparison(self.instance.allocator, stack, geF64),
                0x67 => {
                    const value = try valueAsI32(try popValue(stack));
                    try stack.append(self.instance.allocator, .{ .i32 = @intCast(@clz(value)) });
                },
                0x68 => {
                    const value = try valueAsI32(try popValue(stack));
                    try stack.append(self.instance.allocator, .{ .i32 = @intCast(@ctz(value)) });
                },
                0x69 => {
                    const value = try valueAsI32(try popValue(stack));
                    try stack.append(self.instance.allocator, .{ .i32 = @intCast(@popCount(value)) });
                },
                0x6a => {
                    const rhs = try valueAsI32(try popValue(stack));
                    const lhs = try valueAsI32(try popValue(stack));
                    try stack.append(self.instance.allocator, .{ .i32 = lhs +% rhs });
                },
                0x6b => {
                    const rhs = try valueAsI32(try popValue(stack));
                    const lhs = try valueAsI32(try popValue(stack));
                    try stack.append(self.instance.allocator, .{ .i32 = lhs -% rhs });
                },
                0x6c => {
                    const rhs = try valueAsI32(try popValue(stack));
                    const lhs = try valueAsI32(try popValue(stack));
                    try stack.append(self.instance.allocator, .{ .i32 = lhs *% rhs });
                },
                0x6d => {
                    const rhs: i32 = @bitCast(try valueAsI32(try popValue(stack)));
                    const lhs: i32 = @bitCast(try valueAsI32(try popValue(stack)));
                    if (rhs == 0) return error.IntegerDivideByZero;
                    if (lhs == std.math.minInt(i32) and rhs == -1) return error.IntegerOverflow;
                    try stack.append(self.instance.allocator, .{ .i32 = @bitCast(@divTrunc(lhs, rhs)) });
                },
                0x6e => {
                    const rhs = try valueAsI32(try popValue(stack));
                    const lhs = try valueAsI32(try popValue(stack));
                    if (rhs == 0) return error.IntegerDivideByZero;
                    try stack.append(self.instance.allocator, .{ .i32 = @divTrunc(lhs, rhs) });
                },
                0x6f => {
                    const rhs: i32 = @bitCast(try valueAsI32(try popValue(stack)));
                    const lhs: i32 = @bitCast(try valueAsI32(try popValue(stack)));
                    if (rhs == 0) return error.IntegerDivideByZero;
                    if (lhs == std.math.minInt(i32) and rhs == -1) {
                        try stack.append(self.instance.allocator, .{ .i32 = 0 });
                    } else {
                        try stack.append(self.instance.allocator, .{ .i32 = @bitCast(@rem(lhs, rhs)) });
                    }
                },
                0x70 => {
                    const rhs = try valueAsI32(try popValue(stack));
                    const lhs = try valueAsI32(try popValue(stack));
                    if (rhs == 0) return error.IntegerDivideByZero;
                    try stack.append(self.instance.allocator, .{ .i32 = lhs % rhs });
                },
                0x71 => {
                    const rhs = try valueAsI32(try popValue(stack));
                    const lhs = try valueAsI32(try popValue(stack));
                    try stack.append(self.instance.allocator, .{ .i32 = lhs & rhs });
                },
                0x72 => {
                    const rhs = try valueAsI32(try popValue(stack));
                    const lhs = try valueAsI32(try popValue(stack));
                    try stack.append(self.instance.allocator, .{ .i32 = lhs | rhs });
                },
                0x73 => {
                    const rhs = try valueAsI32(try popValue(stack));
                    const lhs = try valueAsI32(try popValue(stack));
                    try stack.append(self.instance.allocator, .{ .i32 = lhs ^ rhs });
                },
                0x74 => {
                    const shift: u5 = @intCast((try valueAsI32(try popValue(stack))) & 31);
                    const value = try valueAsI32(try popValue(stack));
                    try stack.append(self.instance.allocator, .{ .i32 = value << shift });
                },
                0x75 => {
                    const shift: u5 = @intCast((try valueAsI32(try popValue(stack))) & 31);
                    const value: i32 = @bitCast(try valueAsI32(try popValue(stack)));
                    try stack.append(self.instance.allocator, .{ .i32 = @bitCast(value >> shift) });
                },
                0x76 => {
                    const shift: u5 = @intCast((try valueAsI32(try popValue(stack))) & 31);
                    const value = try valueAsI32(try popValue(stack));
                    try stack.append(self.instance.allocator, .{ .i32 = value >> shift });
                },
                0x77 => {
                    const shift: u5 = @intCast((try valueAsI32(try popValue(stack))) & 31);
                    const value = try valueAsI32(try popValue(stack));
                    try stack.append(self.instance.allocator, .{ .i32 = rotl32(value, shift) });
                },
                0x78 => {
                    const shift: u5 = @intCast((try valueAsI32(try popValue(stack))) & 31);
                    const value = try valueAsI32(try popValue(stack));
                    try stack.append(self.instance.allocator, .{ .i32 = rotr32(value, shift) });
                },
                0x79 => {
                    const value = try valueAsI64(try popValue(stack));
                    try stack.append(self.instance.allocator, .{ .i64 = @intCast(@clz(value)) });
                },
                0x7a => {
                    const value = try valueAsI64(try popValue(stack));
                    try stack.append(self.instance.allocator, .{ .i64 = @intCast(@ctz(value)) });
                },
                0x7b => {
                    const value = try valueAsI64(try popValue(stack));
                    try stack.append(self.instance.allocator, .{ .i64 = @intCast(@popCount(value)) });
                },
                0x7c => {
                    const rhs = try valueAsI64(try popValue(stack));
                    const lhs = try valueAsI64(try popValue(stack));
                    try stack.append(self.instance.allocator, .{ .i64 = lhs +% rhs });
                },
                0x7d => {
                    const rhs = try valueAsI64(try popValue(stack));
                    const lhs = try valueAsI64(try popValue(stack));
                    try stack.append(self.instance.allocator, .{ .i64 = lhs -% rhs });
                },
                0x7e => {
                    const rhs = try valueAsI64(try popValue(stack));
                    const lhs = try valueAsI64(try popValue(stack));
                    try stack.append(self.instance.allocator, .{ .i64 = lhs *% rhs });
                },
                0x7f => {
                    const rhs: i64 = @bitCast(try valueAsI64(try popValue(stack)));
                    const lhs: i64 = @bitCast(try valueAsI64(try popValue(stack)));
                    if (rhs == 0) return error.IntegerDivideByZero;
                    if (lhs == std.math.minInt(i64) and rhs == -1) return error.IntegerOverflow;
                    try stack.append(self.instance.allocator, .{ .i64 = @bitCast(@divTrunc(lhs, rhs)) });
                },
                0x80 => {
                    const rhs = try valueAsI64(try popValue(stack));
                    const lhs = try valueAsI64(try popValue(stack));
                    if (rhs == 0) return error.IntegerDivideByZero;
                    try stack.append(self.instance.allocator, .{ .i64 = @divTrunc(lhs, rhs) });
                },
                0x81 => {
                    const rhs: i64 = @bitCast(try valueAsI64(try popValue(stack)));
                    const lhs: i64 = @bitCast(try valueAsI64(try popValue(stack)));
                    if (rhs == 0) return error.IntegerDivideByZero;
                    if (lhs == std.math.minInt(i64) and rhs == -1) {
                        try stack.append(self.instance.allocator, .{ .i64 = 0 });
                    } else {
                        try stack.append(self.instance.allocator, .{ .i64 = @bitCast(@rem(lhs, rhs)) });
                    }
                },
                0x82 => {
                    const rhs = try valueAsI64(try popValue(stack));
                    const lhs = try valueAsI64(try popValue(stack));
                    if (rhs == 0) return error.IntegerDivideByZero;
                    try stack.append(self.instance.allocator, .{ .i64 = lhs % rhs });
                },
                0x83 => {
                    const rhs = try valueAsI64(try popValue(stack));
                    const lhs = try valueAsI64(try popValue(stack));
                    try stack.append(self.instance.allocator, .{ .i64 = lhs & rhs });
                },
                0x84 => {
                    const rhs = try valueAsI64(try popValue(stack));
                    const lhs = try valueAsI64(try popValue(stack));
                    try stack.append(self.instance.allocator, .{ .i64 = lhs | rhs });
                },
                0x85 => {
                    const rhs = try valueAsI64(try popValue(stack));
                    const lhs = try valueAsI64(try popValue(stack));
                    try stack.append(self.instance.allocator, .{ .i64 = lhs ^ rhs });
                },
                0x86 => {
                    const shift: u6 = @intCast((try valueAsI64(try popValue(stack))) & 63);
                    const value = try valueAsI64(try popValue(stack));
                    try stack.append(self.instance.allocator, .{ .i64 = value << shift });
                },
                0x87 => {
                    const shift: u6 = @intCast((try valueAsI64(try popValue(stack))) & 63);
                    const value: i64 = @bitCast(try valueAsI64(try popValue(stack)));
                    try stack.append(self.instance.allocator, .{ .i64 = @bitCast(value >> shift) });
                },
                0x88 => {
                    const shift: u6 = @intCast((try valueAsI64(try popValue(stack))) & 63);
                    const value = try valueAsI64(try popValue(stack));
                    try stack.append(self.instance.allocator, .{ .i64 = value >> shift });
                },
                0x89 => {
                    const shift: u6 = @intCast((try valueAsI64(try popValue(stack))) & 63);
                    const value = try valueAsI64(try popValue(stack));
                    try stack.append(self.instance.allocator, .{ .i64 = rotl64(value, shift) });
                },
                0x8a => {
                    const shift: u6 = @intCast((try valueAsI64(try popValue(stack))) & 63);
                    const value = try valueAsI64(try popValue(stack));
                    try stack.append(self.instance.allocator, .{ .i64 = rotr64(value, shift) });
                },
                0x92 => {
                    const rhs = try valueAsF32(try popValue(stack));
                    const lhs = try valueAsF32(try popValue(stack));
                    try stack.append(self.instance.allocator, .{ .f32 = lhs + rhs });
                },
                0x93 => {
                    const rhs = try valueAsF32(try popValue(stack));
                    const lhs = try valueAsF32(try popValue(stack));
                    try stack.append(self.instance.allocator, .{ .f32 = lhs - rhs });
                },
                0x94 => {
                    const rhs = try valueAsF32(try popValue(stack));
                    const lhs = try valueAsF32(try popValue(stack));
                    try stack.append(self.instance.allocator, .{ .f32 = lhs * rhs });
                },
                0x95 => {
                    const rhs = try valueAsF32(try popValue(stack));
                    const lhs = try valueAsF32(try popValue(stack));
                    try stack.append(self.instance.allocator, .{ .f32 = lhs / rhs });
                },
                0xa0 => {
                    const rhs = try valueAsF64(try popValue(stack));
                    const lhs = try valueAsF64(try popValue(stack));
                    try stack.append(self.instance.allocator, .{ .f64 = lhs + rhs });
                },
                0xa1 => {
                    const rhs = try valueAsF64(try popValue(stack));
                    const lhs = try valueAsF64(try popValue(stack));
                    try stack.append(self.instance.allocator, .{ .f64 = lhs - rhs });
                },
                0xa2 => {
                    const rhs = try valueAsF64(try popValue(stack));
                    const lhs = try valueAsF64(try popValue(stack));
                    try stack.append(self.instance.allocator, .{ .f64 = lhs * rhs });
                },
                0xa3 => {
                    const rhs = try valueAsF64(try popValue(stack));
                    const lhs = try valueAsF64(try popValue(stack));
                    try stack.append(self.instance.allocator, .{ .f64 = lhs / rhs });
                },
                0xc0 => {
                    const value = try valueAsI32(try popValue(stack));
                    const narrowed: i8 = @bitCast(@as(u8, @intCast(value & 0xff)));
                    const widened: i32 = narrowed;
                    try stack.append(self.instance.allocator, .{ .i32 = @bitCast(widened) });
                },
                0xc1 => {
                    const value = try valueAsI32(try popValue(stack));
                    const narrowed: i16 = @bitCast(@as(u16, @intCast(value & 0xffff)));
                    const widened: i32 = narrowed;
                    try stack.append(self.instance.allocator, .{ .i32 = @bitCast(widened) });
                },
                0xc2 => {
                    const value = try valueAsI64(try popValue(stack));
                    const narrowed: i8 = @bitCast(@as(u8, @intCast(value & 0xff)));
                    const widened: i64 = narrowed;
                    try stack.append(self.instance.allocator, .{ .i64 = @bitCast(widened) });
                },
                0xc3 => {
                    const value = try valueAsI64(try popValue(stack));
                    const narrowed: i16 = @bitCast(@as(u16, @intCast(value & 0xffff)));
                    const widened: i64 = narrowed;
                    try stack.append(self.instance.allocator, .{ .i64 = @bitCast(widened) });
                },
                0xc4 => {
                    const value = try valueAsI64(try popValue(stack));
                    const narrowed: i32 = @bitCast(@as(u32, @intCast(value & 0xffffffff)));
                    const widened: i64 = narrowed;
                    try stack.append(self.instance.allocator, .{ .i64 = @bitCast(widened) });
                },
                else => return error.UnsupportedWasmOpcode,
            }
        }

        return .done;
    }

    fn executeBlock(
        self: *Interpreter,
        reader: *binary.Reader,
        locals: *std.ArrayList(Value),
        stack: *std.ArrayList(Value),
        depth: usize,
    ) anyerror!Flow {
        const boundary = try readBlockBoundary(reader);
        var body_reader = binary.Reader.init(reader.bytes);
        body_reader.offset = boundary.body_start;

        const flow = try self.executeInstructionRange(
            &body_reader,
            boundary.end_offset,
            locals,
            stack,
            depth,
        );

        reader.offset = try std.math.add(usize, boundary.end_offset, 1);
        return consumeBlockBranch(flow);
    }

    fn executeLoop(
        self: *Interpreter,
        reader: *binary.Reader,
        locals: *std.ArrayList(Value),
        stack: *std.ArrayList(Value),
        depth: usize,
    ) anyerror!Flow {
        const boundary = try readBlockBoundary(reader);
        reader.offset = try std.math.add(usize, boundary.end_offset, 1);

        while (true) {
            var body_reader = binary.Reader.init(reader.bytes);
            body_reader.offset = boundary.body_start;

            const flow = try self.executeInstructionRange(
                &body_reader,
                boundary.end_offset,
                locals,
                stack,
                depth,
            );

            switch (flow) {
                .done => return .done,
                .returned => return flow,
                .branch => |label_index| {
                    if (label_index == 0) continue;
                    return .{ .branch = label_index - 1 };
                },
            }
        }
    }

    fn executeIf(
        self: *Interpreter,
        reader: *binary.Reader,
        locals: *std.ArrayList(Value),
        stack: *std.ArrayList(Value),
        depth: usize,
    ) anyerror!Flow {
        const boundary = try readBlockBoundary(reader);
        const next_offset = try std.math.add(usize, boundary.end_offset, 1);
        const condition = try valueAsI32(try popValue(stack));

        reader.offset = next_offset;

        if (condition != 0) {
            var body_reader = binary.Reader.init(reader.bytes);
            body_reader.offset = boundary.body_start;
            const then_limit = boundary.else_offset orelse boundary.end_offset;
            const flow = try self.executeInstructionRange(&body_reader, then_limit, locals, stack, depth);
            return consumeBlockBranch(flow);
        }

        if (boundary.else_offset) |else_offset| {
            var body_reader = binary.Reader.init(reader.bytes);
            body_reader.offset = try std.math.add(usize, else_offset, 1);
            const flow = try self.executeInstructionRange(&body_reader, boundary.end_offset, locals, stack, depth);
            return consumeBlockBranch(flow);
        }

        return .done;
    }

    fn callFunctionFromStack(
        self: *Interpreter,
        function_index: u32,
        stack: *std.ArrayList(Value),
        depth: usize,
    ) !void {
        const imported_count = self.instance.importedFunctionCount();
        const actual = std.math.cast(usize, function_index) orelse return error.InvalidFunctionIndex;

        if (actual < imported_count) {
            const imported = try self.instance.importedFunction(function_index);
            const function_type = try self.instance.importedFunctionType(function_index);
            const arity = function_type.params.len;
            var args_buffer: [7]Value = undefined;

            if (arity > args_buffer.len) return error.UnsupportedImportArity;

            var remaining = arity;
            while (remaining != 0) {
                remaining -= 1;
                args_buffer[remaining] = try popValue(stack);
            }

            const result = try self.callImportWithValues(imported, args_buffer[0..arity]);
            if (try resultFromImport(function_type, result)) |value| {
                try stack.append(self.instance.allocator, value);
            }
            return;
        }

        const function = try self.instance.definedFunction(function_index);
        const function_type = try self.instance.functionType(function.type_index);
        const arity = function_type.params.len;
        var args_buffer: [16]Value = undefined;

        if (arity > args_buffer.len) return error.UnsupportedFunctionArity;

        var remaining = arity;
        while (remaining != 0) {
            remaining -= 1;
            args_buffer[remaining] = try popValue(stack);
        }

        const result = try self.executeFunction(function_index, args_buffer[0..arity], depth);
        if (result) |value| {
            try stack.append(self.instance.allocator, value);
        }
    }

    fn callImportWithValues(
        self: *Interpreter,
        function: imports.Function,
        args: []const Value,
    ) !u32 {
        var args_buffer: [7]u32 = undefined;

        if (args.len > args_buffer.len) return error.UnsupportedImportArity;

        for (args, 0..) |arg, index| {
            args_buffer[index] = try valueAsI32(arg);
        }

        return self.callImport(function, args_buffer[0..args.len]);
    }
};

fn readLocalDeclarations(
    allocator: std.mem.Allocator,
    reader: *binary.Reader,
    locals: *std.ArrayList(Value),
) !void {
    const group_count = try reader.readVarU32();

    for (0..group_count) |_| {
        const count = try reader.readVarU32();
        const value_type = try reader.readByte();

        for (0..count) |_| {
            switch (value_type) {
                0x7f => try locals.append(allocator, .{ .i32 = 0 }),
                0x7e => try locals.append(allocator, .{ .i64 = 0 }),
                0x7d => try locals.append(allocator, .{ .f32 = 0 }),
                0x7c => try locals.append(allocator, .{ .f64 = 0 }),
                else => return error.UnsupportedLocalType,
            }
        }
    }
}

fn validateFunctionArgs(function_type: module.FunctionType, args: []const Value) !void {
    if (args.len != function_type.params.len) return error.FunctionArityMismatch;

    for (args, function_type.params) |arg, param_type| {
        if (!valueMatchesType(arg, param_type)) return error.FunctionArgumentTypeMismatch;
    }
}

fn validateFunctionResult(function_type: module.FunctionType, result: ?Value) !void {
    if (function_type.results.len > 1) return error.UnsupportedMultiValueResult;

    if (function_type.results.len == 0) {
        if (result != null) return error.UnexpectedFunctionResult;
        return;
    }

    const actual = result orelse return error.MissingFunctionResult;
    if (!valueMatchesType(actual, function_type.results[0])) return error.FunctionResultTypeMismatch;
}

fn resultFromImport(function_type: module.FunctionType, result: u32) !?Value {
    if (function_type.results.len > 1) return error.UnsupportedMultiValueResult;
    if (function_type.results.len == 0) return null;
    if (function_type.results[0] != .i32) return error.UnsupportedImportResultType;

    return .{ .i32 = result };
}

fn valueMatchesType(value: Value, value_type: module.ValueType) bool {
    return switch (value_type) {
        .i32 => std.meta.activeTag(value) == .i32,
        .i64 => std.meta.activeTag(value) == .i64,
        .f32 => std.meta.activeTag(value) == .f32,
        .f64 => std.meta.activeTag(value) == .f64,
    };
}

fn valueFromConst(value: module.ConstValue) Value {
    return switch (value) {
        .i32 => |actual| .{ .i32 = actual },
        .i64 => |actual| .{ .i64 = actual },
        .f32 => |actual| .{ .f32 = actual },
        .f64 => |actual| .{ .f64 = actual },
    };
}

fn constValueFromValue(value: Value, value_type: module.ValueType) !module.ConstValue {
    return switch (value_type) {
        .i32 => .{ .i32 = try valueAsI32(value) },
        .i64 => .{ .i64 = try valueAsI64(value) },
        .f32 => switch (value) {
            .f32 => |actual| .{ .f32 = actual },
            else => error.ExpectedF32Value,
        },
        .f64 => switch (value) {
            .f64 => |actual| .{ .f64 = actual },
            else => error.ExpectedF64Value,
        },
    };
}

fn localValue(locals: []const Value, index: u32) !Value {
    const actual = std.math.cast(usize, index) orelse return error.InvalidLocalIndex;
    if (actual >= locals.len) return error.InvalidLocalIndex;

    return locals[actual];
}

fn setLocalValue(locals: []Value, index: u32, value: Value) !void {
    const actual = std.math.cast(usize, index) orelse return error.InvalidLocalIndex;
    if (actual >= locals.len) return error.InvalidLocalIndex;

    locals[actual] = value;
}

fn readBranchTableTarget(reader: *binary.Reader, stack: *std.ArrayList(Value)) !u32 {
    const label_count = try reader.readVarU32();
    const label_count_usize = std.math.cast(usize, label_count) orelse return error.BranchTableTooLarge;
    const target_index = try valueAsI32(try popValue(stack));
    var selected: ?u32 = null;

    for (0..label_count_usize) |index| {
        const label_index = try reader.readVarU32();
        const index_u32 = std.math.cast(u32, index) orelse return error.BranchTableTooLarge;
        if (target_index == index_u32) selected = label_index;
    }

    const default_label = try reader.readVarU32();
    return selected orelse default_label;
}

fn readSelectTypeVector(reader: *binary.Reader) !void {
    const count = try reader.readVarU32();
    if (count != 1) return error.UnsupportedSelectTypeVector;

    switch (try reader.readByte()) {
        0x7f, 0x7e, 0x7d, 0x7c => {},
        else => return error.UnsupportedValueType,
    }
}

fn pushSelectedValue(allocator: std.mem.Allocator, stack: *std.ArrayList(Value)) !void {
    const condition = try valueAsI32(try popValue(stack));
    const on_false = try popValue(stack);
    const on_true = try popValue(stack);

    if (std.meta.activeTag(on_true) != std.meta.activeTag(on_false)) {
        return error.SelectTypeMismatch;
    }

    try stack.append(allocator, if (condition != 0) on_true else on_false);
}

fn readBlockBoundary(reader: *binary.Reader) !BlockBoundary {
    try readBlockType(reader);

    const body_start = reader.offset;
    const boundary = try findBlockBoundary(reader.bytes, body_start);

    return .{
        .body_start = body_start,
        .else_offset = boundary.else_offset,
        .end_offset = boundary.end_offset,
    };
}

fn findBlockBoundary(bytes: []const u8, body_start: usize) !BlockBoundary {
    var reader = binary.Reader.init(bytes);
    reader.offset = body_start;
    var depth: usize = 0;
    var else_offset: ?usize = null;

    while (reader.offset < bytes.len) {
        const opcode_offset = reader.offset;
        const opcode = try reader.readByte();

        switch (opcode) {
            0x02, 0x03, 0x04 => {
                try readBlockType(&reader);
                depth += 1;
            },
            0x05 => {
                if (depth == 0) {
                    if (else_offset != null) return error.MultipleElseBranches;
                    else_offset = opcode_offset;
                }
            },
            0x0b => {
                if (depth == 0) {
                    return .{
                        .body_start = body_start,
                        .else_offset = else_offset,
                        .end_offset = opcode_offset,
                    };
                }

                depth -= 1;
            },
            else => try skipInstructionImmediate(&reader, opcode),
        }
    }

    return error.UnterminatedBlock;
}

fn readBlockType(reader: *binary.Reader) !void {
    const block_type = try reader.readByte();

    switch (block_type) {
        0x40, 0x7f, 0x7e, 0x7d, 0x7c => {},
        else => return error.UnsupportedBlockType,
    }
}

fn skipInstructionImmediate(reader: *binary.Reader, opcode: u8) !void {
    switch (opcode) {
        0x00, 0x01, 0x0f, 0x1a, 0x1b, 0x45...0xc4 => {},
        0x0c, 0x0d, 0x10, 0x20, 0x21, 0x22, 0x23, 0x24 => _ = try reader.readVarU32(),
        0x0e => {
            const label_count = try reader.readVarU32();
            for (0..label_count) |_| {
                _ = try reader.readVarU32();
            }
            _ = try reader.readVarU32();
        },
        0x11 => {
            _ = try reader.readVarU32();
            _ = try reader.readVarU32();
        },
        0x1c => try readSelectTypeVector(reader),
        0x28...0x3e => {
            _ = try reader.readVarU32();
            _ = try reader.readVarU32();
        },
        0x3f, 0x40 => {
            const memory_index = try reader.readByte();
            if (memory_index != 0) return error.UnsupportedMemoryIndex;
        },
        0x41 => _ = try reader.readVarI32(),
        0x42 => _ = try reader.readVarI64(),
        0x43 => _ = try reader.readBytes(4),
        0x44 => _ = try reader.readBytes(8),
        else => return error.UnsupportedWasmOpcode,
    }
}

fn consumeBlockBranch(flow: Flow) Flow {
    return switch (flow) {
        .branch => |label_index| if (label_index == 0) .done else .{ .branch = label_index - 1 },
        else => flow,
    };
}

fn popOptional(stack: *std.ArrayList(Value)) ?Value {
    if (stack.items.len == 0) return null;
    const value = stack.items[stack.items.len - 1];
    stack.items.len -= 1;
    return value;
}

fn popValue(stack: *std.ArrayList(Value)) !Value {
    return popOptional(stack) orelse error.StackUnderflow;
}

fn peekValue(stack: *std.ArrayList(Value)) !Value {
    if (stack.items.len == 0) return error.StackUnderflow;
    return stack.items[stack.items.len - 1];
}

fn valueAsI32(value: Value) !u32 {
    return switch (value) {
        .i32 => |actual| actual,
        else => error.ExpectedI32Value,
    };
}

fn valueAsI64(value: Value) !u64 {
    return switch (value) {
        .i64 => |actual| actual,
        else => error.ExpectedI64Value,
    };
}

fn valueAsF32(value: Value) !f32 {
    return switch (value) {
        .f32 => |actual| actual,
        else => error.ExpectedF32Value,
    };
}

fn valueAsF64(value: Value) !f64 {
    return switch (value) {
        .f64 => |actual| actual,
        else => error.ExpectedF64Value,
    };
}

fn pushI32Comparison(
    allocator: std.mem.Allocator,
    stack: *std.ArrayList(Value),
    predicate: *const fn (u32, u32) bool,
) !void {
    const rhs = try valueAsI32(try popValue(stack));
    const lhs = try valueAsI32(try popValue(stack));

    try stack.append(allocator, .{ .i32 = if (predicate(lhs, rhs)) 1 else 0 });
}

fn pushI32SignedComparison(
    allocator: std.mem.Allocator,
    stack: *std.ArrayList(Value),
    predicate: *const fn (i32, i32) bool,
) !void {
    const rhs: i32 = @bitCast(try valueAsI32(try popValue(stack)));
    const lhs: i32 = @bitCast(try valueAsI32(try popValue(stack)));

    try stack.append(allocator, .{ .i32 = if (predicate(lhs, rhs)) 1 else 0 });
}

fn pushI64Comparison(
    allocator: std.mem.Allocator,
    stack: *std.ArrayList(Value),
    predicate: *const fn (u64, u64) bool,
) !void {
    const rhs = try valueAsI64(try popValue(stack));
    const lhs = try valueAsI64(try popValue(stack));

    try stack.append(allocator, .{ .i32 = if (predicate(lhs, rhs)) 1 else 0 });
}

fn pushI64SignedComparison(
    allocator: std.mem.Allocator,
    stack: *std.ArrayList(Value),
    predicate: *const fn (i64, i64) bool,
) !void {
    const rhs: i64 = @bitCast(try valueAsI64(try popValue(stack)));
    const lhs: i64 = @bitCast(try valueAsI64(try popValue(stack)));

    try stack.append(allocator, .{ .i32 = if (predicate(lhs, rhs)) 1 else 0 });
}

fn pushF32Comparison(
    allocator: std.mem.Allocator,
    stack: *std.ArrayList(Value),
    predicate: *const fn (f32, f32) bool,
) !void {
    const rhs = try valueAsF32(try popValue(stack));
    const lhs = try valueAsF32(try popValue(stack));

    try stack.append(allocator, .{ .i32 = if (predicate(lhs, rhs)) 1 else 0 });
}

fn pushF64Comparison(
    allocator: std.mem.Allocator,
    stack: *std.ArrayList(Value),
    predicate: *const fn (f64, f64) bool,
) !void {
    const rhs = try valueAsF64(try popValue(stack));
    const lhs = try valueAsF64(try popValue(stack));

    try stack.append(allocator, .{ .i32 = if (predicate(lhs, rhs)) 1 else 0 });
}

fn eqU32(lhs: u32, rhs: u32) bool {
    return lhs == rhs;
}

fn neU32(lhs: u32, rhs: u32) bool {
    return lhs != rhs;
}

fn ltU32(lhs: u32, rhs: u32) bool {
    return lhs < rhs;
}

fn gtU32(lhs: u32, rhs: u32) bool {
    return lhs > rhs;
}

fn leU32(lhs: u32, rhs: u32) bool {
    return lhs <= rhs;
}

fn geU32(lhs: u32, rhs: u32) bool {
    return lhs >= rhs;
}

fn neU64(lhs: u64, rhs: u64) bool {
    return lhs != rhs;
}

fn ltU64(lhs: u64, rhs: u64) bool {
    return lhs < rhs;
}

fn gtU64(lhs: u64, rhs: u64) bool {
    return lhs > rhs;
}

fn leU64(lhs: u64, rhs: u64) bool {
    return lhs <= rhs;
}

fn geU64(lhs: u64, rhs: u64) bool {
    return lhs >= rhs;
}

fn ltI32(lhs: i32, rhs: i32) bool {
    return lhs < rhs;
}

fn gtI32(lhs: i32, rhs: i32) bool {
    return lhs > rhs;
}

fn leI32(lhs: i32, rhs: i32) bool {
    return lhs <= rhs;
}

fn geI32(lhs: i32, rhs: i32) bool {
    return lhs >= rhs;
}

fn ltI64(lhs: i64, rhs: i64) bool {
    return lhs < rhs;
}

fn gtI64(lhs: i64, rhs: i64) bool {
    return lhs > rhs;
}

fn leI64(lhs: i64, rhs: i64) bool {
    return lhs <= rhs;
}

fn geI64(lhs: i64, rhs: i64) bool {
    return lhs >= rhs;
}

fn eqF32(lhs: f32, rhs: f32) bool {
    return lhs == rhs;
}

fn neF32(lhs: f32, rhs: f32) bool {
    return lhs != rhs;
}

fn ltF32(lhs: f32, rhs: f32) bool {
    return lhs < rhs;
}

fn gtF32(lhs: f32, rhs: f32) bool {
    return lhs > rhs;
}

fn leF32(lhs: f32, rhs: f32) bool {
    return lhs <= rhs;
}

fn geF32(lhs: f32, rhs: f32) bool {
    return lhs >= rhs;
}

fn eqF64(lhs: f64, rhs: f64) bool {
    return lhs == rhs;
}

fn neF64(lhs: f64, rhs: f64) bool {
    return lhs != rhs;
}

fn ltF64(lhs: f64, rhs: f64) bool {
    return lhs < rhs;
}

fn gtF64(lhs: f64, rhs: f64) bool {
    return lhs > rhs;
}

fn leF64(lhs: f64, rhs: f64) bool {
    return lhs <= rhs;
}

fn geF64(lhs: f64, rhs: f64) bool {
    return lhs >= rhs;
}

fn rotl32(value: u32, shift: u5) u32 {
    if (shift == 0) return value;
    return (value << shift) | (value >> @intCast(32 - @as(u6, shift)));
}

fn rotr32(value: u32, shift: u5) u32 {
    if (shift == 0) return value;
    return (value >> shift) | (value << @intCast(32 - @as(u6, shift)));
}

fn rotl64(value: u64, shift: u6) u64 {
    if (shift == 0) return value;
    return (value << shift) | (value >> @intCast(64 - @as(u7, shift)));
}

fn rotr64(value: u64, shift: u6) u64 {
    if (shift == 0) return value;
    return (value >> shift) | (value << @intCast(64 - @as(u7, shift)));
}

test "interpreter executes exported function returning i32" {
    const allocator = std.testing.allocator;

    var parsed = try module.Module.parse(allocator, fixtures.return_i32_seven);
    defer parsed.deinit(allocator);

    var wasm_instance = try instance.Instance.init(allocator, &parsed, 64 * 1024);
    defer wasm_instance.deinit();

    var interpreter = Interpreter.init(&wasm_instance);
    const result = (try interpreter.callExport("run", &.{})) orelse return error.MissingReturnValue;

    try std.testing.expectEqual(@as(u32, 7), try valueAsI32(result));
}

test "interpreter executes basic i32 arithmetic and comparisons" {
    const allocator = std.testing.allocator;

    var parsed = try module.Module.parse(allocator, fixtures.i32_integer_ops);
    defer parsed.deinit(allocator);

    var wasm_instance = try instance.Instance.init(allocator, &parsed, 64 * 1024);
    defer wasm_instance.deinit();

    var interpreter = Interpreter.init(&wasm_instance);
    const result = (try interpreter.callExport("run", &.{})) orelse return error.MissingReturnValue;

    try std.testing.expectEqual(@as(u32, 1), try valueAsI32(result));
}

test "interpreter executes integer memory ops" {
    const allocator = std.testing.allocator;

    var parsed = try module.Module.parse(allocator, fixtures.memory_integer_ops);
    defer parsed.deinit(allocator);

    var wasm_instance = try instance.Instance.init(allocator, &parsed, 64 * 1024);
    defer wasm_instance.deinit();

    var interpreter = Interpreter.init(&wasm_instance);
    const result = (try interpreter.callExport("run", &.{})) orelse return error.MissingReturnValue;

    try std.testing.expectEqual(@as(u32, 2), try valueAsI32(result));
    try std.testing.expectEqual(@as(u64, 28), readU64Little(try wasm_instance.memory.read(40, 8)));
    try std.testing.expectEqual(@as(u8, 255), (try wasm_instance.memory.read(48, 1))[0]);
}

test "interpreter decodes signed i32 constants" {
    const allocator = std.testing.allocator;

    var parsed = try module.Module.parse(allocator, fixtures.signed_i32_const);
    defer parsed.deinit(allocator);

    var wasm_instance = try instance.Instance.init(allocator, &parsed, 64 * 1024);
    defer wasm_instance.deinit();

    var interpreter = Interpreter.init(&wasm_instance);
    const result = (try interpreter.callExport("run", &.{})) orelse return error.MissingReturnValue;

    try std.testing.expectEqual(@as(u32, 0xffffffff), try valueAsI32(result));
}

test "interpreter executes if else branching" {
    const allocator = std.testing.allocator;

    var parsed = try module.Module.parse(allocator, fixtures.if_else_branching);
    defer parsed.deinit(allocator);

    var wasm_instance = try instance.Instance.init(allocator, &parsed, 64 * 1024);
    defer wasm_instance.deinit();

    var interpreter = Interpreter.init(&wasm_instance);
    const result = (try interpreter.callExport("run", &.{})) orelse return error.MissingReturnValue;

    try std.testing.expectEqual(@as(u32, 42), try valueAsI32(result));
}

test "interpreter executes loops and conditional branches" {
    const allocator = std.testing.allocator;

    var parsed = try module.Module.parse(allocator, fixtures.loop_sum_to_five);
    defer parsed.deinit(allocator);

    var wasm_instance = try instance.Instance.init(allocator, &parsed, 64 * 1024);
    defer wasm_instance.deinit();

    var interpreter = Interpreter.init(&wasm_instance);
    const result = (try interpreter.callExport("run", &.{})) orelse return error.MissingReturnValue;

    try std.testing.expectEqual(@as(u32, 10), try valueAsI32(result));
}

test "interpreter exits on return instruction" {
    const allocator = std.testing.allocator;

    var parsed = try module.Module.parse(allocator, fixtures.early_return);
    defer parsed.deinit(allocator);

    var wasm_instance = try instance.Instance.init(allocator, &parsed, 64 * 1024);
    defer wasm_instance.deinit();

    var interpreter = Interpreter.init(&wasm_instance);
    const result = (try interpreter.callExport("run", &.{})) orelse return error.MissingReturnValue;

    try std.testing.expectEqual(@as(u32, 9), try valueAsI32(result));
}

test "interpreter passes arguments to defined functions" {
    const allocator = std.testing.allocator;

    var parsed = try module.Module.parse(allocator, fixtures.defined_function_call);
    defer parsed.deinit(allocator);

    var wasm_instance = try instance.Instance.init(allocator, &parsed, 64 * 1024);
    defer wasm_instance.deinit();

    var interpreter = Interpreter.init(&wasm_instance);
    const result = (try interpreter.callExport("run", &.{})) orelse return error.MissingReturnValue;

    try std.testing.expectEqual(@as(u32, 15), try valueAsI32(result));
}

test "interpreter executes extended i32 numeric ops" {
    const allocator = std.testing.allocator;

    var parsed = try module.Module.parse(allocator, fixtures.i32_numeric_ops_extended);
    defer parsed.deinit(allocator);

    var wasm_instance = try instance.Instance.init(allocator, &parsed, 64 * 1024);
    defer wasm_instance.deinit();

    var interpreter = Interpreter.init(&wasm_instance);
    const result = (try interpreter.callExport("run", &.{})) orelse return error.MissingReturnValue;

    try std.testing.expectEqual(@as(u32, 43), try valueAsI32(result));
}

test "interpreter executes i32 16-bit memory ops" {
    const allocator = std.testing.allocator;

    var parsed = try module.Module.parse(allocator, fixtures.i32_memory16_ops);
    defer parsed.deinit(allocator);

    var wasm_instance = try instance.Instance.init(allocator, &parsed, 64 * 1024);
    defer wasm_instance.deinit();

    var interpreter = Interpreter.init(&wasm_instance);
    const result = (try interpreter.callExport("run", &.{})) orelse return error.MissingReturnValue;

    try std.testing.expectEqual(@as(u32, 2), try valueAsI32(result));
    try std.testing.expectEqual(@as(u16, 0xff80), readU16Little(try wasm_instance.memory.read(128, 2)));
}

test "interpreter executes select and branch table" {
    const allocator = std.testing.allocator;

    var parsed = try module.Module.parse(allocator, fixtures.control_select_and_br_table);
    defer parsed.deinit(allocator);

    var wasm_instance = try instance.Instance.init(allocator, &parsed, 64 * 1024);
    defer wasm_instance.deinit();

    var interpreter = Interpreter.init(&wasm_instance);
    const result = (try interpreter.callExport("run", &.{})) orelse return error.MissingReturnValue;

    try std.testing.expectEqual(@as(u32, 7), try valueAsI32(result));
}

test "interpreter executes i64 and float numeric ops" {
    const allocator = std.testing.allocator;

    var parsed = try module.Module.parse(allocator, fixtures.numeric_i64_and_float_ops);
    defer parsed.deinit(allocator);

    var wasm_instance = try instance.Instance.init(allocator, &parsed, 64 * 1024);
    defer wasm_instance.deinit();

    var interpreter = Interpreter.init(&wasm_instance);
    const result = (try interpreter.callExport("run", &.{})) orelse return error.MissingReturnValue;

    try std.testing.expectEqual(@as(u32, 3), try valueAsI32(result));
}

test "interpreter executes extended i64 memory ops" {
    const allocator = std.testing.allocator;

    var parsed = try module.Module.parse(allocator, fixtures.extended_i64_memory_ops);
    defer parsed.deinit(allocator);

    var wasm_instance = try instance.Instance.init(allocator, &parsed, 64 * 1024);
    defer wasm_instance.deinit();

    var interpreter = Interpreter.init(&wasm_instance);
    const result = (try interpreter.callExport("run", &.{})) orelse return error.MissingReturnValue;

    try std.testing.expectEqual(@as(u32, 2), try valueAsI32(result));
    try std.testing.expectEqual(@as(u8, 254), (try wasm_instance.memory.read(0, 1))[0]);
    try std.testing.expectEqual(@as(u32, 255), try wasm_instance.memory.readU32(8));
}

test "interpreter executes globals and start function" {
    const allocator = std.testing.allocator;

    var parsed = try module.Module.parse(allocator, fixtures.globals_and_start);
    defer parsed.deinit(allocator);

    var wasm_instance = try instance.Instance.init(allocator, &parsed, 64 * 1024);
    defer wasm_instance.deinit();

    var interpreter = Interpreter.init(&wasm_instance);
    const before_start = (try interpreter.callExport("run", &.{})) orelse return error.MissingReturnValue;

    try std.testing.expectEqual(@as(u32, 7), try valueAsI32(before_start));

    try interpreter.runStart();

    const after_start = (try interpreter.callExport("run", &.{})) orelse return error.MissingReturnValue;
    try std.testing.expectEqual(@as(u32, 12), try valueAsI32(after_start));

    try interpreter.runStart();

    const after_second_start = (try interpreter.callExport("run", &.{})) orelse return error.MissingReturnValue;
    try std.testing.expectEqual(@as(u32, 12), try valueAsI32(after_second_start));
}

test "interpreter executes memory size and grow" {
    const allocator = std.testing.allocator;

    var parsed = try module.Module.parse(allocator, fixtures.memory_size_and_grow);
    defer parsed.deinit(allocator);

    var wasm_instance = try instance.Instance.init(allocator, &parsed, 64 * 1024);
    defer wasm_instance.deinit();

    var interpreter = Interpreter.init(&wasm_instance);
    const result = (try interpreter.callExport("run", &.{})) orelse return error.MissingReturnValue;

    try std.testing.expectEqual(@as(u32, 3), try valueAsI32(result));
    try std.testing.expectEqual(@as(u32, 2), try wasm_instance.currentMemoryPages());
}

test "interpreter routes wasi fd_write import" {
    const allocator = std.testing.allocator;

    var parsed = try module.Module.parse(allocator, fixtures.wasi_fd_write);
    defer parsed.deinit(allocator);

    var wasm_instance = try instance.Instance.init(allocator, &parsed, 64 * 1024);
    defer wasm_instance.deinit();

    var resolver = imports.Resolver.initWasi(allocator, &wasm_instance.memory);
    defer resolver.deinit();
    try wasm_instance.bindImports(&resolver);

    var interpreter = Interpreter.init(&wasm_instance);
    const result = (try interpreter.callExport("run", &.{})) orelse return error.MissingReturnValue;

    try std.testing.expectEqual(@as(u32, 0), try valueAsI32(result));
    try std.testing.expectEqual(@as(u32, 5), try wasm_instance.memory.readU32(16));
    try std.testing.expectEqualStrings("hello", resolver.stdout.items);
}

test "interpreter routes wasi-nn import calls through ABI resolver" {
    const allocator = std.testing.allocator;
    const wasi_nn_abi = @import("wasi_nn_abi");

    var parsed = try module.Module.parse(allocator, fixtures.wasi_nn_compute_smoke);
    defer parsed.deinit(allocator);

    var wasm_instance = try instance.Instance.init(allocator, &parsed, 64 * 1024);
    defer wasm_instance.deinit();

    var host = wasi_nn_abi.Host.init(allocator);
    defer host.deinit();

    var surface = wasi_nn_abi.Surface.init(allocator, &host, &wasm_instance.memory);
    var resolver = imports.Resolver.init(&surface);
    try wasm_instance.bindImports(&resolver);

    var interpreter = Interpreter.init(&wasm_instance);
    const result = (try interpreter.callExport("run", &.{})) orelse return error.MissingReturnValue;

    try std.testing.expectEqual(
        @intFromEnum(wasi_nn_abi.Status.invalid_context_handle),
        try valueAsI32(result),
    );
}

test "interpreter runs tiny mnist wasi-nn flow from guest memory" {
    const allocator = std.testing.allocator;
    const wasi_nn_abi = @import("wasi_nn_abi");

    const model_bytes = try std.Io.Dir.cwd().readFileAlloc(
        std.Options.debug_io,
        "models/tiny_mnist.onnx",
        allocator,
        .limited(100 * 1024 * 1024),
    );
    defer allocator.free(model_bytes);

    var parsed = try module.Module.parse(allocator, fixtures.wasi_nn_tiny_mnist_flow);
    defer parsed.deinit(allocator);

    var wasm_instance = try instance.Instance.init(allocator, &parsed, 2 * 1024 * 1024);
    defer wasm_instance.deinit();

    try wasm_instance.memory.write(1024, model_bytes);
    try std.testing.expectEqual(@as(u64, 1), readU64Little(try wasm_instance.memory.read(200000, 8)));
    try std.testing.expectEqual(@as(u64, 28), readU64Little(try wasm_instance.memory.read(200016, 8)));

    var host = wasi_nn_abi.Host.init(allocator);
    defer host.deinit();

    var surface = wasi_nn_abi.Surface.init(allocator, &host, &wasm_instance.memory);
    var resolver = imports.Resolver.init(&surface);
    try wasm_instance.bindImports(&resolver);

    var interpreter = Interpreter.init(&wasm_instance);
    const result = (try interpreter.callExport("run", &.{
        .{ .i32 = std.math.cast(u32, model_bytes.len) orelse return error.ModelTooLarge },
    })) orelse return error.MissingReturnValue;

    try std.testing.expectEqual(@intFromEnum(wasi_nn_abi.Status.ok), try valueAsI32(result));

    const context_handle = try wasm_instance.memory.readU32(20);
    const output = try host.getOutput(.{ .index = context_handle }, 0);

    try std.testing.expectEqualStrings("probabilities", output.name);
    try std.testing.expectEqualSlices(usize, &.{ 1, 10 }, output.value.shape);
}

test "interpreter runs constrained edge inference guest flow" {
    const allocator = std.testing.allocator;
    const wasi_nn_abi = @import("wasi_nn_abi");

    const model_bytes = try std.Io.Dir.cwd().readFileAlloc(
        std.Options.debug_io,
        "models/tiny_mnist.onnx",
        allocator,
        .limited(100 * 1024 * 1024),
    );
    defer allocator.free(model_bytes);

    var parsed = try module.Module.parse(allocator, fixtures.constrained_edge_inference_flow);
    defer parsed.deinit(allocator);

    var wasm_instance = try instance.Instance.init(allocator, &parsed, 64 * 1024);
    defer wasm_instance.deinit();

    try wasm_instance.memory.write(1024, model_bytes);

    var host = wasi_nn_abi.Host.init(allocator);
    defer host.deinit();

    var surface = wasi_nn_abi.Surface.init(allocator, &host, &wasm_instance.memory);
    var resolver = imports.Resolver.init(&surface);
    defer resolver.deinit();
    try wasm_instance.bindImports(&resolver);

    var interpreter = Interpreter.init(&wasm_instance);
    const result = (try interpreter.callExport("run", &.{
        .{ .i32 = std.math.cast(u32, model_bytes.len) orelse return error.ModelTooLarge },
    })) orelse return error.MissingReturnValue;

    try std.testing.expectEqual(@intFromEnum(wasi_nn_abi.Status.ok), try valueAsI32(result));
    try std.testing.expectEqual(@as(u32, 2), try wasm_instance.currentMemoryPages());
    try std.testing.expectEqual(@as(u32, 10), try wasm_instance.memory.readU32(16));
    try std.testing.expectEqualStrings("edge:start\nedge:done\n", resolver.stdout.items);

    try std.testing.expectEqual(@intFromEnum(wasi_nn_abi.DType.float32), try wasm_instance.memory.readU32(28));
    try std.testing.expectEqual(@as(u32, 2), try wasm_instance.memory.readU32(36));
    try std.testing.expectEqual(@as(u32, 40), try wasm_instance.memory.readU32(40));
    try std.testing.expectEqual(@as(u32, 40), try wasm_instance.memory.readU32(44));
    try std.testing.expectEqual(@as(u64, 1), readU64Little(try wasm_instance.memory.read(96, 8)));
    try std.testing.expectEqual(@as(u64, 10), readU64Little(try wasm_instance.memory.read(104, 8)));

    const context_handle = try wasm_instance.memory.readU32(24);
    const output = try host.getOutput(.{ .index = context_handle }, 0);
    const output_values = try output.value.float32Data();

    try std.testing.expectEqualStrings("probabilities", output.name);
    try std.testing.expectEqualSlices(usize, &.{ 1, 10 }, output.value.shape);
    try std.testing.expectEqual(@as(u32, @bitCast(output_values[0])), try wasm_instance.memory.readU32(128));
    try std.testing.expectEqual(@as(u32, @bitCast(output_values[9])), try wasm_instance.memory.readU32(164));
}

fn readU64Little(bytes: []const u8) u64 {
    return @as(u64, bytes[0]) |
        (@as(u64, bytes[1]) << 8) |
        (@as(u64, bytes[2]) << 16) |
        (@as(u64, bytes[3]) << 24) |
        (@as(u64, bytes[4]) << 32) |
        (@as(u64, bytes[5]) << 40) |
        (@as(u64, bytes[6]) << 48) |
        (@as(u64, bytes[7]) << 56);
}

fn readU32Little(bytes: []const u8) u32 {
    return @as(u32, bytes[0]) |
        (@as(u32, bytes[1]) << 8) |
        (@as(u32, bytes[2]) << 16) |
        (@as(u32, bytes[3]) << 24);
}

fn readU16Little(bytes: []const u8) u16 {
    return @as(u16, bytes[0]) |
        (@as(u16, bytes[1]) << 8);
}

fn writeU16Little(bytes: []u8, value: u16) void {
    bytes[0] = std.math.cast(u8, value & 0xff) orelse unreachable;
    bytes[1] = std.math.cast(u8, (value >> 8) & 0xff) orelse unreachable;
}

fn writeU64Little(bytes: []u8, value: u64) void {
    bytes[0] = std.math.cast(u8, value & 0xff) orelse unreachable;
    bytes[1] = std.math.cast(u8, (value >> 8) & 0xff) orelse unreachable;
    bytes[2] = std.math.cast(u8, (value >> 16) & 0xff) orelse unreachable;
    bytes[3] = std.math.cast(u8, (value >> 24) & 0xff) orelse unreachable;
    bytes[4] = std.math.cast(u8, (value >> 32) & 0xff) orelse unreachable;
    bytes[5] = std.math.cast(u8, (value >> 40) & 0xff) orelse unreachable;
    bytes[6] = std.math.cast(u8, (value >> 48) & 0xff) orelse unreachable;
    bytes[7] = std.math.cast(u8, (value >> 56) & 0xff) orelse unreachable;
}
