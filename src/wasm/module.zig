const std = @import("std");
const binary = @import("binary.zig");

const section_count_len = 13;

pub const Limits = struct {
    min: u32,
    max: ?u32 = null,
};

pub const ImportKind = enum {
    function,
    table,
    memory,
    global,
};

pub const Import = struct {
    module: []const u8,
    name: []const u8,
    kind: ImportKind,
    type_index: ?u32 = null,
    limits: ?Limits = null,
};

pub const ExportKind = enum {
    function,
    table,
    memory,
    global,
};

pub const Export = struct {
    name: []const u8,
    kind: ExportKind,
    index: u32,
};

pub const ValueType = enum(u8) {
    i32 = 0x7f,
    i64 = 0x7e,
    f32 = 0x7d,
    f64 = 0x7c,
};

pub const FunctionType = struct {
    params: []const ValueType,
    results: []const ValueType,
};

pub const ConstValue = union(enum) {
    i32: u32,
    i64: u64,
    f32: f32,
    f64: f64,
};

pub const Function = struct {
    type_index: u32,
    body: []const u8,
};

pub const GlobalType = struct {
    value_type: ValueType,
    mutable: bool,
};

pub const Global = struct {
    global_type: GlobalType,
    init: ConstValue,
};

pub const Memory = struct {
    limits: Limits,
};

pub const DataSegment = struct {
    memory_index: u32 = 0,
    offset: u32,
    bytes: []const u8,
};

pub const Module = struct {
    bytes: []const u8,
    section_counts: [section_count_len]usize = [_]usize{0} ** section_count_len,
    function_types: std.ArrayList(FunctionType) = .empty,
    imports: std.ArrayList(Import) = .empty,
    exports: std.ArrayList(Export) = .empty,
    functions: std.ArrayList(Function) = .empty,
    globals: std.ArrayList(Global) = .empty,
    memories: std.ArrayList(Memory) = .empty,
    data_segments: std.ArrayList(DataSegment) = .empty,
    start_function_index: ?u32 = null,

    pub fn parse(allocator: std.mem.Allocator, bytes: []const u8) !Module {
        var reader = binary.Reader.init(bytes);
        try reader.readHeader();

        var result = Module{
            .bytes = bytes,
        };
        errdefer result.deinit(allocator);

        var function_type_indices: std.ArrayList(u32) = .empty;
        defer function_type_indices.deinit(allocator);

        while (try reader.nextSection()) |section| {
            result.section_counts[@intFromEnum(section.id)] += 1;

            switch (section.id) {
                .type => try parseTypeSection(allocator, section.payload, &result),
                .import => try parseImportSection(allocator, section.payload, &result),
                .function => try parseFunctionSection(allocator, section.payload, &function_type_indices),
                .memory => try parseMemorySection(allocator, section.payload, &result),
                .global => try parseGlobalSection(allocator, section.payload, &result),
                .@"export" => try parseExportSection(allocator, section.payload, &result),
                .start => try parseStartSection(section.payload, &result),
                .code => try parseCodeSection(allocator, section.payload, function_type_indices.items, &result),
                .data => try parseDataSection(allocator, section.payload, &result),
                else => {},
            }
        }

        if (function_type_indices.items.len != result.functions.items.len) {
            return error.FunctionCodeCountMismatch;
        }

        return result;
    }

    pub fn deinit(self: *Module, allocator: std.mem.Allocator) void {
        self.function_types.deinit(allocator);
        self.imports.deinit(allocator);
        self.exports.deinit(allocator);
        self.functions.deinit(allocator);
        self.globals.deinit(allocator);
        self.memories.deinit(allocator);
        self.data_segments.deinit(allocator);
        self.* = undefined;
    }

    pub fn sectionCount(self: Module, id: binary.SectionId) usize {
        return self.section_counts[@intFromEnum(id)];
    }

    pub fn functionType(self: Module, type_index: u32) !FunctionType {
        const actual = std.math.cast(usize, type_index) orelse return error.InvalidFunctionTypeIndex;
        if (actual >= self.function_types.items.len) return error.InvalidFunctionTypeIndex;

        return self.function_types.items[actual];
    }
};

fn parseTypeSection(allocator: std.mem.Allocator, payload: []const u8, parsed: *Module) !void {
    var reader = binary.Reader.init(payload);
    const count = try reader.readVarU32();

    for (0..count) |_| {
        const form = try reader.readByte();
        if (form != 0x60) return error.UnsupportedFunctionTypeForm;

        const params = try readValueTypeVector(&reader);
        const results = try readValueTypeVector(&reader);

        try parsed.function_types.append(allocator, .{
            .params = params,
            .results = results,
        });
    }

    try expectSectionEnd(reader);
}

fn parseImportSection(allocator: std.mem.Allocator, payload: []const u8, parsed: *Module) !void {
    var reader = binary.Reader.init(payload);
    const count = try reader.readVarU32();

    for (0..count) |_| {
        const module_name = try readName(&reader);
        const name = try readName(&reader);
        const kind_byte = try reader.readByte();

        const import = switch (kind_byte) {
            0x00 => Import{
                .module = module_name,
                .name = name,
                .kind = .function,
                .type_index = try reader.readVarU32(),
            },
            0x01 => blk: {
                try skipTableType(&reader);
                break :blk Import{
                    .module = module_name,
                    .name = name,
                    .kind = .table,
                };
            },
            0x02 => Import{
                .module = module_name,
                .name = name,
                .kind = .memory,
                .limits = try readLimits(&reader),
            },
            0x03 => blk: {
                try skipGlobalType(&reader);
                break :blk Import{
                    .module = module_name,
                    .name = name,
                    .kind = .global,
                };
            },
            else => return error.UnsupportedImportKind,
        };

        try parsed.imports.append(allocator, import);
    }

    try expectSectionEnd(reader);
}

fn parseFunctionSection(
    allocator: std.mem.Allocator,
    payload: []const u8,
    function_type_indices: *std.ArrayList(u32),
) !void {
    var reader = binary.Reader.init(payload);
    const count = try reader.readVarU32();

    for (0..count) |_| {
        try function_type_indices.append(allocator, try reader.readVarU32());
    }

    try expectSectionEnd(reader);
}

fn parseMemorySection(allocator: std.mem.Allocator, payload: []const u8, parsed: *Module) !void {
    var reader = binary.Reader.init(payload);
    const count = try reader.readVarU32();

    for (0..count) |_| {
        try parsed.memories.append(allocator, .{
            .limits = try readLimits(&reader),
        });
    }

    try expectSectionEnd(reader);
}

fn parseGlobalSection(allocator: std.mem.Allocator, payload: []const u8, parsed: *Module) !void {
    var reader = binary.Reader.init(payload);
    const count = try reader.readVarU32();

    for (0..count) |_| {
        const global_type = try readGlobalType(&reader);
        const init = try readConstExpr(&reader, global_type.value_type);

        try parsed.globals.append(allocator, .{
            .global_type = global_type,
            .init = init,
        });
    }

    try expectSectionEnd(reader);
}

fn parseExportSection(allocator: std.mem.Allocator, payload: []const u8, parsed: *Module) !void {
    var reader = binary.Reader.init(payload);
    const count = try reader.readVarU32();

    for (0..count) |_| {
        const name = try readName(&reader);
        const kind = try exportKindFromByte(try reader.readByte());
        const index = try reader.readVarU32();

        try parsed.exports.append(allocator, .{
            .name = name,
            .kind = kind,
            .index = index,
        });
    }

    try expectSectionEnd(reader);
}

fn parseStartSection(payload: []const u8, parsed: *Module) !void {
    var reader = binary.Reader.init(payload);
    parsed.start_function_index = try reader.readVarU32();

    try expectSectionEnd(reader);
}

fn parseCodeSection(
    allocator: std.mem.Allocator,
    payload: []const u8,
    function_type_indices: []const u32,
    parsed: *Module,
) !void {
    var reader = binary.Reader.init(payload);
    const count = try reader.readVarU32();
    const count_usize = std.math.cast(usize, count) orelse return error.FunctionCodeCountMismatch;

    if (count_usize != function_type_indices.len) return error.FunctionCodeCountMismatch;

    for (function_type_indices) |type_index| {
        const body_size = try reader.readVarU32();
        const body = try reader.readBytes(body_size);

        try parsed.functions.append(allocator, .{
            .type_index = type_index,
            .body = body,
        });
    }

    try expectSectionEnd(reader);
}

fn parseDataSection(allocator: std.mem.Allocator, payload: []const u8, parsed: *Module) !void {
    var reader = binary.Reader.init(payload);
    const count = try reader.readVarU32();

    for (0..count) |_| {
        const mode = try reader.readVarU32();

        const segment = switch (mode) {
            0 => DataSegment{
                .memory_index = 0,
                .offset = try readI32ConstExpr(&reader),
                .bytes = try readName(&reader),
            },
            1 => return error.PassiveDataSegmentUnsupported,
            2 => blk: {
                const memory_index = try reader.readVarU32();
                break :blk DataSegment{
                    .memory_index = memory_index,
                    .offset = try readI32ConstExpr(&reader),
                    .bytes = try readName(&reader),
                };
            },
            else => return error.UnsupportedDataSegmentMode,
        };

        try parsed.data_segments.append(allocator, segment);
    }

    try expectSectionEnd(reader);
}

fn readName(reader: *binary.Reader) ![]const u8 {
    const len = try reader.readVarU32();
    return reader.readBytes(len);
}

fn readValueTypeVector(reader: *binary.Reader) ![]const ValueType {
    const len = try reader.readVarU32();
    const bytes = try reader.readBytes(len);

    for (bytes) |byte| {
        _ = try valueTypeFromByte(byte);
    }

    return @ptrCast(bytes);
}

fn valueTypeFromByte(value: u8) !ValueType {
    return switch (value) {
        0x7f => .i32,
        0x7e => .i64,
        0x7d => .f32,
        0x7c => .f64,
        else => error.UnsupportedValueType,
    };
}

fn readGlobalType(reader: *binary.Reader) !GlobalType {
    const value_type = try valueTypeFromByte(try reader.readByte());
    const mutability = try reader.readByte();

    return switch (mutability) {
        0 => .{
            .value_type = value_type,
            .mutable = false,
        },
        1 => .{
            .value_type = value_type,
            .mutable = true,
        },
        else => error.UnsupportedGlobalType,
    };
}

fn readConstExpr(reader: *binary.Reader, expected_type: ValueType) !ConstValue {
    const opcode = try reader.readByte();

    const value: ConstValue = switch (opcode) {
        0x41 => blk: {
            if (expected_type != .i32) return error.InvalidConstExpressionType;
            break :blk .{ .i32 = @bitCast(try reader.readVarI32()) };
        },
        0x42 => blk: {
            if (expected_type != .i64) return error.InvalidConstExpressionType;
            break :blk .{ .i64 = @bitCast(try reader.readVarI64()) };
        },
        0x43 => blk: {
            if (expected_type != .f32) return error.InvalidConstExpressionType;
            break :blk .{ .f32 = @bitCast(readU32Little(try reader.readBytes(4))) };
        },
        0x44 => blk: {
            if (expected_type != .f64) return error.InvalidConstExpressionType;
            break :blk .{ .f64 = @bitCast(readU64Little(try reader.readBytes(8))) };
        },
        else => return error.UnsupportedConstExpression,
    };

    const end = try reader.readByte();
    if (end != 0x0b) return error.InvalidConstExpression;

    return value;
}

fn readI32ConstExpr(reader: *binary.Reader) !u32 {
    const value = try readConstExpr(reader, .i32);

    return switch (value) {
        .i32 => |actual| if ((actual & 0x80000000) == 0) actual else error.InvalidOffsetExpression,
        else => error.InvalidOffsetExpression,
    };
}

fn readLimits(reader: *binary.Reader) !Limits {
    const flags = try reader.readVarU32();
    const min = try reader.readVarU32();

    return switch (flags) {
        0 => .{
            .min = min,
        },
        1 => .{
            .min = min,
            .max = try reader.readVarU32(),
        },
        else => error.UnsupportedLimits,
    };
}

fn skipTableType(reader: *binary.Reader) !void {
    const element_type = try reader.readByte();
    if (element_type != 0x70) return error.UnsupportedTableType;

    _ = try readLimits(reader);
}

fn skipGlobalType(reader: *binary.Reader) !void {
    _ = try readGlobalType(reader);
}

fn exportKindFromByte(value: u8) !ExportKind {
    return switch (value) {
        0x00 => .function,
        0x01 => .table,
        0x02 => .memory,
        0x03 => .global,
        else => error.UnsupportedExportKind,
    };
}

fn expectSectionEnd(reader: binary.Reader) !void {
    if (!reader.isAtEnd()) return error.TrailingSectionBytes;
}

fn readU32Little(bytes: []const u8) u32 {
    return @as(u32, bytes[0]) |
        (@as(u32, bytes[1]) << 8) |
        (@as(u32, bytes[2]) << 16) |
        (@as(u32, bytes[3]) << 24);
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

test "module parse records section counts" {
    const allocator = std.testing.allocator;

    var parsed = try Module.parse(allocator, "\x00asm\x01\x00\x00\x00\x01\x01\x00");
    defer parsed.deinit(allocator);

    try std.testing.expectEqual(@as(usize, 1), parsed.sectionCount(.type));
    try std.testing.expectEqual(@as(usize, 0), parsed.sectionCount(.import));
}

test "module parse decodes function imports" {
    const allocator = std.testing.allocator;
    const bytes =
        "\x00asm\x01\x00\x00\x00" ++
        "\x02\x13" ++
        "\x01" ++
        "\x07wasi_nn" ++
        "\x07compute" ++
        "\x00" ++
        "\x00";

    var parsed = try Module.parse(allocator, bytes);
    defer parsed.deinit(allocator);

    try std.testing.expectEqual(@as(usize, 1), parsed.imports.items.len);
    try std.testing.expectEqualStrings("wasi_nn", parsed.imports.items[0].module);
    try std.testing.expectEqualStrings("compute", parsed.imports.items[0].name);
    try std.testing.expectEqual(ImportKind.function, parsed.imports.items[0].kind);
    try std.testing.expectEqual(@as(?u32, 0), parsed.imports.items[0].type_index);
}

test "module parse decodes function types" {
    const allocator = std.testing.allocator;
    const bytes =
        "\x00asm\x01\x00\x00\x00" ++
        "\x01\x0b" ++
        "\x02\x60\x02\x7f\x7f\x01\x7f\x60\x00\x01\x7e";

    var parsed = try Module.parse(allocator, bytes);
    defer parsed.deinit(allocator);

    try std.testing.expectEqual(@as(usize, 2), parsed.function_types.items.len);
    try std.testing.expectEqualSlices(ValueType, &.{ .i32, .i32 }, parsed.function_types.items[0].params);
    try std.testing.expectEqualSlices(ValueType, &.{.i32}, parsed.function_types.items[0].results);
    try std.testing.expectEqual(@as(usize, 0), parsed.function_types.items[1].params.len);
    try std.testing.expectEqualSlices(ValueType, &.{.i64}, parsed.function_types.items[1].results);
}

test "module parse decodes globals and start function" {
    const allocator = std.testing.allocator;
    const bytes =
        "\x00asm\x01\x00\x00\x00" ++
        "\x06\x06" ++
        "\x01\x7f\x01\x41\x07\x0b" ++
        "\x08\x01" ++
        "\x00";

    var parsed = try Module.parse(allocator, bytes);
    defer parsed.deinit(allocator);

    try std.testing.expectEqual(@as(usize, 1), parsed.sectionCount(.global));
    try std.testing.expectEqual(@as(usize, 1), parsed.sectionCount(.start));
    try std.testing.expectEqual(@as(usize, 1), parsed.globals.items.len);
    try std.testing.expectEqual(ValueType.i32, parsed.globals.items[0].global_type.value_type);
    try std.testing.expect(parsed.globals.items[0].global_type.mutable);
    try std.testing.expectEqual(@as(?u32, 0), parsed.start_function_index);
    try std.testing.expectEqual(@as(u32, 7), switch (parsed.globals.items[0].init) {
        .i32 => |value| value,
        else => return error.ExpectedI32Global,
    });
}

test "module parse decodes float globals" {
    const allocator = std.testing.allocator;
    const bytes =
        "\x00asm\x01\x00\x00\x00" ++
        "\x06\x15" ++
        "\x02" ++
        "\x7d\x00\x43\x00\x00\xc0\x3f\x0b" ++
        "\x7c\x01\x44\x00\x00\x00\x00\x00\x00\x00\x40\x0b";

    var parsed = try Module.parse(allocator, bytes);
    defer parsed.deinit(allocator);

    try std.testing.expectEqual(@as(usize, 2), parsed.globals.items.len);
    try std.testing.expectEqual(ValueType.f32, parsed.globals.items[0].global_type.value_type);
    try std.testing.expect(!parsed.globals.items[0].global_type.mutable);
    try std.testing.expectEqual(@as(u32, 0x3fc00000), switch (parsed.globals.items[0].init) {
        .f32 => |value| @as(u32, @bitCast(value)),
        else => return error.ExpectedF32Global,
    });

    try std.testing.expectEqual(ValueType.f64, parsed.globals.items[1].global_type.value_type);
    try std.testing.expect(parsed.globals.items[1].global_type.mutable);
    try std.testing.expectEqual(@as(u64, 0x4000000000000000), switch (parsed.globals.items[1].init) {
        .f64 => |value| @as(u64, @bitCast(value)),
        else => return error.ExpectedF64Global,
    });
}

test "module parse decodes memory and exports" {
    const allocator = std.testing.allocator;
    const bytes =
        "\x00asm\x01\x00\x00\x00" ++
        "\x05\x03" ++
        "\x01\x00\x01" ++
        "\x07\x0a" ++
        "\x01" ++
        "\x06memory" ++
        "\x02" ++
        "\x00";

    var parsed = try Module.parse(allocator, bytes);
    defer parsed.deinit(allocator);

    try std.testing.expectEqual(@as(usize, 1), parsed.memories.items.len);
    try std.testing.expectEqual(@as(u32, 1), parsed.memories.items[0].limits.min);
    try std.testing.expect(parsed.memories.items[0].limits.max == null);

    try std.testing.expectEqual(@as(usize, 1), parsed.exports.items.len);
    try std.testing.expectEqualStrings("memory", parsed.exports.items[0].name);
    try std.testing.expectEqual(ExportKind.memory, parsed.exports.items[0].kind);
    try std.testing.expectEqual(@as(u32, 0), parsed.exports.items[0].index);
}

test "module parse decodes function bodies" {
    const allocator = std.testing.allocator;
    const bytes =
        "\x00asm\x01\x00\x00\x00" ++
        "\x01\x04" ++
        "\x01\x60\x00\x00" ++
        "\x03\x02" ++
        "\x01\x00" ++
        "\x07\x07" ++
        "\x01\x03run\x00\x00" ++
        "\x0a\x04" ++
        "\x01\x02\x00\x0b";

    var parsed = try Module.parse(allocator, bytes);
    defer parsed.deinit(allocator);

    try std.testing.expectEqual(@as(usize, 1), parsed.functions.items.len);
    try std.testing.expectEqual(@as(u32, 0), parsed.functions.items[0].type_index);
    try std.testing.expectEqualSlices(u8, "\x00\x0b", parsed.functions.items[0].body);

    try std.testing.expectEqual(@as(usize, 1), parsed.exports.items.len);
    try std.testing.expectEqualStrings("run", parsed.exports.items[0].name);
    try std.testing.expectEqual(ExportKind.function, parsed.exports.items[0].kind);
    try std.testing.expectEqual(@as(u32, 0), parsed.exports.items[0].index);
}

test "module parse decodes active data segments" {
    const allocator = std.testing.allocator;
    const bytes =
        "\x00asm\x01\x00\x00\x00" ++
        "\x05\x03" ++
        "\x01\x00\x01" ++
        "\x0b\x09" ++
        "\x01\x00\x41\x04\x0b\x03abc";

    var parsed = try Module.parse(allocator, bytes);
    defer parsed.deinit(allocator);

    try std.testing.expectEqual(@as(usize, 1), parsed.data_segments.items.len);
    try std.testing.expectEqual(@as(u32, 0), parsed.data_segments.items[0].memory_index);
    try std.testing.expectEqual(@as(u32, 4), parsed.data_segments.items[0].offset);
    try std.testing.expectEqualSlices(u8, "abc", parsed.data_segments.items[0].bytes);
}
