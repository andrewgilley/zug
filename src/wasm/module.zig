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

pub const Function = struct {
    type_index: u32,
    body: []const u8,
};

pub const Memory = struct {
    limits: Limits,
};

pub const Module = struct {
    bytes: []const u8,
    section_counts: [section_count_len]usize = [_]usize{0} ** section_count_len,
    imports: std.ArrayList(Import) = .empty,
    exports: std.ArrayList(Export) = .empty,
    functions: std.ArrayList(Function) = .empty,
    memories: std.ArrayList(Memory) = .empty,

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
                .import => try parseImportSection(allocator, section.payload, &result),
                .function => try parseFunctionSection(allocator, section.payload, &function_type_indices),
                .memory => try parseMemorySection(allocator, section.payload, &result),
                .@"export" => try parseExportSection(allocator, section.payload, &result),
                .code => try parseCodeSection(allocator, section.payload, function_type_indices.items, &result),
                else => {},
            }
        }

        if (function_type_indices.items.len != result.functions.items.len) {
            return error.FunctionCodeCountMismatch;
        }

        return result;
    }

    pub fn deinit(self: *Module, allocator: std.mem.Allocator) void {
        self.imports.deinit(allocator);
        self.exports.deinit(allocator);
        self.functions.deinit(allocator);
        self.memories.deinit(allocator);
        self.* = undefined;
    }

    pub fn sectionCount(self: Module, id: binary.SectionId) usize {
        return self.section_counts[@intFromEnum(id)];
    }
};

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

fn readName(reader: *binary.Reader) ![]const u8 {
    const len = try reader.readVarU32();
    return reader.readBytes(len);
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
    const value_type = try reader.readByte();
    switch (value_type) {
        0x7f, 0x7e, 0x7d, 0x7c => {},
        else => return error.UnsupportedGlobalType,
    }

    const mutability = try reader.readByte();
    if (mutability != 0 and mutability != 1) return error.UnsupportedGlobalType;
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

test "module parse records section counts" {
    const allocator = std.testing.allocator;

    var parsed = try Module.parse(allocator, "\x00asm\x01\x00\x00\x00\x01\x00");
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
