const std = @import("std");
const binary = @import("binary.zig");

pub const version: u16 = 0x000d;
pub const layer: u16 = 0x0001;

pub const SectionId = enum(u8) {
    custom = 0,
    core_module = 1,
    core_instance = 2,
    core_type = 3,
    component = 4,
    instance = 5,
    alias = 6,
    type = 7,
    canon = 8,
    start = 9,
    import = 10,
    @"export" = 11,
    value = 12,
};

pub const Section = struct {
    id: SectionId,
    payload: []const u8,
};

pub const CoreSort = enum(u8) {
    function = 0x00,
    table = 0x01,
    memory = 0x02,
    global = 0x03,
    tag = 0x04,
    type = 0x10,
    module = 0x11,
    instance = 0x12,
};

pub const Sort = union(enum) {
    core: CoreSort,
    function,
    value,
    type,
    component,
    instance,
};

pub const TypeBound = union(enum) {
    eq: u32,
    sub_resource,
};

pub const ValueBound = union(enum) {
    eq: u32,
    type_index: u32,
};

pub const ExternDesc = union(enum) {
    core_module_type: u32,
    function_type: u32,
    value: ValueBound,
    type: TypeBound,
    component_type: u32,
    instance_type: u32,
};

pub const SortIndex = struct {
    sort: Sort,
    index: u32,
};

pub const Import = struct {
    name: []const u8,
    version: ?[]const u8 = null,
    desc: ExternDesc,
};

pub const Export = struct {
    name: []const u8,
    version: ?[]const u8 = null,
    sort_index: SortIndex,
    desc: ?ExternDesc = null,
};

pub const Component = struct {
    bytes: []const u8,
    section_counts: [13]usize = [_]usize{0} ** 13,
    imports: std.ArrayList(Import) = .empty,
    exports: std.ArrayList(Export) = .empty,

    pub fn parse(allocator: std.mem.Allocator, bytes: []const u8) !Component {
        var reader = Reader.init(bytes);
        try reader.readHeader();

        var result = Component{
            .bytes = bytes,
        };

        while (try reader.nextSection()) |section| {
            result.section_counts[@intFromEnum(section.id)] += 1;

            switch (section.id) {
                .import => try parseImportSection(allocator, section.payload, &result.imports),
                .@"export" => try parseExportSection(allocator, section.payload, &result.exports),
                else => {},
            }
        }

        return result;
    }

    pub fn deinit(self: *Component, allocator: std.mem.Allocator) void {
        self.imports.deinit(allocator);
        self.exports.deinit(allocator);
        self.* = undefined;
    }

    pub fn sectionCount(self: Component, id: SectionId) usize {
        return self.section_counts[@intFromEnum(id)];
    }
};

pub const Reader = struct {
    bytes: []const u8,
    offset: usize = 0,

    pub fn init(bytes: []const u8) Reader {
        return .{
            .bytes = bytes,
        };
    }

    pub fn readHeader(self: *Reader) !void {
        if (self.bytes.len < 8) return error.InvalidWasmHeader;
        if (!std.mem.eql(u8, self.bytes[0..4], "\x00asm")) return error.InvalidWasmMagic;

        const actual_version = readU16Little(self.bytes[4..6]);
        const actual_layer = readU16Little(self.bytes[6..8]);

        if (actual_version != version) return error.UnsupportedComponentVersion;
        if (actual_layer != layer) return error.UnsupportedComponentLayer;

        self.offset = 8;
    }

    pub fn nextSection(self: *Reader) !?Section {
        if (self.offset == self.bytes.len) return null;
        if (self.offset > self.bytes.len) return error.InvalidSectionRange;

        const id = try sectionIdFromByte(try self.readByte());
        const payload_len = try self.readVarU32();
        const payload_start = self.offset;
        const payload_len_usize = std.math.cast(usize, payload_len) orelse return error.SectionTooLarge;
        const payload_end = try std.math.add(usize, payload_start, payload_len_usize);

        if (payload_end > self.bytes.len) return error.InvalidSectionRange;

        self.offset = payload_end;

        return .{
            .id = id,
            .payload = self.bytes[payload_start..payload_end],
        };
    }

    pub fn readVarU32(self: *Reader) !u32 {
        var core_reader = binary.Reader{
            .bytes = self.bytes,
            .offset = self.offset,
        };
        const value = try core_reader.readVarU32();
        self.offset = core_reader.offset;
        return value;
    }

    pub fn readBytes(self: *Reader, len: u32) ![]const u8 {
        const start = self.offset;
        const byte_len = std.math.cast(usize, len) orelse return error.InvalidByteRange;
        const end = try std.math.add(usize, start, byte_len);

        if (end > self.bytes.len) return error.UnexpectedEndOfWasm;

        self.offset = end;

        return self.bytes[start..end];
    }

    pub fn isAtEnd(self: Reader) bool {
        return self.offset == self.bytes.len;
    }

    pub fn readByte(self: *Reader) !u8 {
        if (self.offset >= self.bytes.len) return error.UnexpectedEndOfWasm;
        const byte = self.bytes[self.offset];
        self.offset += 1;
        return byte;
    }
};

pub fn isComponent(bytes: []const u8) bool {
    if (bytes.len < 8) return false;
    return std.mem.eql(u8, bytes[0..4], "\x00asm") and
        readU16Little(bytes[4..6]) == version and
        readU16Little(bytes[6..8]) == layer;
}

pub fn sectionName(id: SectionId) []const u8 {
    return switch (id) {
        .custom => "custom",
        .core_module => "core-module",
        .core_instance => "core-instance",
        .core_type => "core-type",
        .component => "component",
        .instance => "instance",
        .alias => "alias",
        .type => "type",
        .canon => "canon",
        .start => "start",
        .import => "import",
        .@"export" => "export",
        .value => "value",
    };
}

pub fn sortName(sort: Sort) []const u8 {
    return switch (sort) {
        .core => |core| switch (core) {
            .function => "core-func",
            .table => "core-table",
            .memory => "core-memory",
            .global => "core-global",
            .tag => "core-tag",
            .type => "core-type",
            .module => "core-module",
            .instance => "core-instance",
        },
        .function => "func",
        .value => "value",
        .type => "type",
        .component => "component",
        .instance => "instance",
    };
}

pub fn descName(desc: ExternDesc) []const u8 {
    return switch (desc) {
        .core_module_type => "core-module",
        .function_type => "func",
        .value => "value",
        .type => "type",
        .component_type => "component",
        .instance_type => "instance",
    };
}

pub fn descTypeIndex(desc: ExternDesc) ?u32 {
    return switch (desc) {
        .core_module_type => |index| index,
        .function_type => |index| index,
        .value => |bound| switch (bound) {
            .eq => |index| index,
            .type_index => |index| index,
        },
        .type => |bound| switch (bound) {
            .eq => |index| index,
            .sub_resource => null,
        },
        .component_type => |index| index,
        .instance_type => |index| index,
    };
}

fn parseImportSection(allocator: std.mem.Allocator, payload: []const u8, imports: *std.ArrayList(Import)) !void {
    var reader = Reader.init(payload);
    const count = try reader.readVarU32();

    for (0..count) |_| {
        const name = try readImportExportName(&reader);
        const desc = try readExternDesc(&reader);
        try imports.append(allocator, .{
            .name = name.name,
            .version = name.version,
            .desc = desc,
        });
    }

    try expectEnd(reader);
}

fn parseExportSection(allocator: std.mem.Allocator, payload: []const u8, exports: *std.ArrayList(Export)) !void {
    var reader = Reader.init(payload);
    const count = try reader.readVarU32();

    for (0..count) |_| {
        const name = try readImportExportName(&reader);
        const sort_index = try readSortIndex(&reader);
        const desc = try readOptionalExternDesc(&reader);

        try exports.append(allocator, .{
            .name = name.name,
            .version = name.version,
            .sort_index = sort_index,
            .desc = desc,
        });
    }

    try expectEnd(reader);
}

const Name = struct {
    name: []const u8,
    version: ?[]const u8 = null,
};

fn readImportExportName(reader: *Reader) !Name {
    const tag = try reader.readByte();
    const name = try readName(reader);

    return switch (tag) {
        0x00, 0x01 => .{ .name = name },
        0x02 => .{
            .name = name,
            .version = try readName(reader),
        },
        else => error.UnsupportedComponentNameKind,
    };
}

fn readName(reader: *Reader) ![]const u8 {
    const len = try reader.readVarU32();
    return reader.readBytes(len);
}

fn readOptionalExternDesc(reader: *Reader) !?ExternDesc {
    const tag = try reader.readByte();
    return switch (tag) {
        0x00 => null,
        0x01 => try readExternDesc(reader),
        else => error.UnsupportedOptionalExternDesc,
    };
}

fn readExternDesc(reader: *Reader) !ExternDesc {
    const tag = try reader.readByte();
    return switch (tag) {
        0x00 => blk: {
            const core_sort = try readCoreSort(reader);
            if (core_sort != .module) return error.UnsupportedComponentExternDesc;
            break :blk .{ .core_module_type = try reader.readVarU32() };
        },
        0x01 => .{ .function_type = try reader.readVarU32() },
        0x02 => .{ .value = try readValueBound(reader) },
        0x03 => .{ .type = try readTypeBound(reader) },
        0x04 => .{ .component_type = try reader.readVarU32() },
        0x05 => .{ .instance_type = try reader.readVarU32() },
        else => error.UnsupportedComponentExternDesc,
    };
}

fn readSortIndex(reader: *Reader) !SortIndex {
    return .{
        .sort = try readSort(reader),
        .index = try reader.readVarU32(),
    };
}

fn readSort(reader: *Reader) !Sort {
    const tag = try reader.readByte();
    return switch (tag) {
        0x00 => .{ .core = try readCoreSort(reader) },
        0x01 => .function,
        0x02 => .value,
        0x03 => .type,
        0x04 => .component,
        0x05 => .instance,
        else => error.UnsupportedComponentSort,
    };
}

fn readCoreSort(reader: *Reader) !CoreSort {
    return switch (try reader.readByte()) {
        0x00 => .function,
        0x01 => .table,
        0x02 => .memory,
        0x03 => .global,
        0x04 => .tag,
        0x10 => .type,
        0x11 => .module,
        0x12 => .instance,
        else => error.UnsupportedComponentCoreSort,
    };
}

fn readValueBound(reader: *Reader) !ValueBound {
    const tag = try reader.readByte();
    return switch (tag) {
        0x00 => .{ .eq = try reader.readVarU32() },
        0x01 => .{ .type_index = try reader.readVarU32() },
        else => error.UnsupportedComponentValueBound,
    };
}

fn readTypeBound(reader: *Reader) !TypeBound {
    const tag = try reader.readByte();
    return switch (tag) {
        0x00 => .{ .eq = try reader.readVarU32() },
        0x01 => .sub_resource,
        else => error.UnsupportedComponentTypeBound,
    };
}

fn expectEnd(reader: Reader) !void {
    if (!reader.isAtEnd()) return error.TrailingComponentSectionBytes;
}

fn sectionIdFromByte(value: u8) !SectionId {
    return switch (value) {
        0 => .custom,
        1 => .core_module,
        2 => .core_instance,
        3 => .core_type,
        4 => .component,
        5 => .instance,
        6 => .alias,
        7 => .type,
        8 => .canon,
        9 => .start,
        10 => .import,
        11 => .@"export",
        12 => .value,
        else => error.UnsupportedComponentSection,
    };
}

fn readU16Little(bytes: []const u8) u16 {
    return @as(u16, bytes[0]) | (@as(u16, bytes[1]) << 8);
}

test "component parser accepts empty component" {
    var parsed = try Component.parse(std.testing.allocator, "\x00asm\x0d\x00\x01\x00");
    defer parsed.deinit(std.testing.allocator);

    try std.testing.expectEqual(@as(usize, 0), parsed.sectionCount(.type));
    try std.testing.expectEqual(@as(usize, 0), parsed.sectionCount(.@"export"));
}

test "component parser counts sections" {
    var parsed = try Component.parse(
        std.testing.allocator,
        "\x00asm\x0d\x00\x01\x00" ++
            "\x07\x01\x00" ++
            "\x0b\x01\x00",
    );
    defer parsed.deinit(std.testing.allocator);

    try std.testing.expectEqual(@as(usize, 1), parsed.sectionCount(.type));
    try std.testing.expectEqual(@as(usize, 1), parsed.sectionCount(.@"export"));
}

test "component parser rejects core module header" {
    try std.testing.expect(!isComponent("\x00asm\x01\x00\x00\x00"));
    try std.testing.expectError(error.UnsupportedComponentVersion, Component.parse(std.testing.allocator, "\x00asm\x01\x00\x00\x00"));
}

test "component parser decodes imports and exports" {
    var parsed = try Component.parse(
        std.testing.allocator,
        "\x00asm\x0d\x00\x01\x00" ++
            "\x0a\x08" ++
            "\x01" ++
            "\x00\x03env" ++
            "\x01\x00" ++
            "\x0b\x0b" ++
            "\x01" ++
            "\x00\x03run" ++
            "\x01\x00" ++
            "\x01\x01\x00",
    );
    defer parsed.deinit(std.testing.allocator);

    try std.testing.expectEqual(@as(usize, 1), parsed.imports.items.len);
    try std.testing.expectEqualStrings("env", parsed.imports.items[0].name);
    try std.testing.expectEqualStrings("func", descName(parsed.imports.items[0].desc));
    try std.testing.expectEqual(@as(?u32, 0), descTypeIndex(parsed.imports.items[0].desc));

    try std.testing.expectEqual(@as(usize, 1), parsed.exports.items.len);
    try std.testing.expectEqualStrings("run", parsed.exports.items[0].name);
    try std.testing.expectEqualStrings("func", sortName(parsed.exports.items[0].sort_index.sort));
    try std.testing.expectEqual(@as(u32, 0), parsed.exports.items[0].sort_index.index);
    try std.testing.expectEqualStrings("func", descName(parsed.exports.items[0].desc.?));
    try std.testing.expectEqual(@as(?u32, 0), descTypeIndex(parsed.exports.items[0].desc.?));
}
