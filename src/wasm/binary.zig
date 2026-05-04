const std = @import("std");

pub const version = 1;

pub const SectionId = enum(u8) {
    custom = 0,
    type = 1,
    import = 2,
    function = 3,
    table = 4,
    memory = 5,
    global = 6,
    @"export" = 7,
    start = 8,
    element = 9,
    code = 10,
    data = 11,
    data_count = 12,
};

pub const Section = struct {
    id: SectionId,
    payload: []const u8,
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

        const actual_version =
            @as(u32, self.bytes[4]) |
            (@as(u32, self.bytes[5]) << 8) |
            (@as(u32, self.bytes[6]) << 16) |
            (@as(u32, self.bytes[7]) << 24);

        if (actual_version != version) return error.UnsupportedWasmVersion;

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
        var result: u32 = 0;
        var shift: u5 = 0;

        while (true) {
            const byte = try self.readByte();
            result |= @as(u32, byte & 0x7f) << shift;

            if ((byte & 0x80) == 0) return result;
            if (shift >= 28) return error.InvalidVarInt;
            shift += 7;
        }
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

fn sectionIdFromByte(value: u8) !SectionId {
    return switch (value) {
        0 => .custom,
        1 => .type,
        2 => .import,
        3 => .function,
        4 => .table,
        5 => .memory,
        6 => .global,
        7 => .@"export",
        8 => .start,
        9 => .element,
        10 => .code,
        11 => .data,
        12 => .data_count,
        else => error.UnsupportedSection,
    };
}

test "reader accepts an empty wasm module" {
    var reader = Reader.init("\x00asm\x01\x00\x00\x00");

    try reader.readHeader();
    try std.testing.expect((try reader.nextSection()) == null);
}

test "reader returns section payload slices" {
    var reader = Reader.init("\x00asm\x01\x00\x00\x00\x01\x00");

    try reader.readHeader();

    const section = (try reader.nextSection()) orelse return error.MissingSection;
    try std.testing.expectEqual(SectionId.type, section.id);
    try std.testing.expectEqual(@as(usize, 0), section.payload.len);
    try std.testing.expect((try reader.nextSection()) == null);
}
