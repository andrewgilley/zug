const std = @import("std");

pub const Package = struct {
    id: []const u8,
    interfaces: []const Interface = &.{},
    worlds: []const World = &.{},
};

pub const Interface = struct {
    name: []const u8,
    uses: []const Use = &.{},
    declarations: []const Declaration = &.{},
    functions: []const Function = &.{},
};

pub const Use = struct {
    interface_name: []const u8,
    names: []const []const u8,
};

pub const Declaration = union(enum) {
    alias: Alias,
    record: Record,
    @"enum": Enum,
    variant: Variant,
};

pub const Alias = struct {
    name: []const u8,
    target: Type,
};

pub const Record = struct {
    name: []const u8,
    fields: []const Field,
};

pub const Field = struct {
    name: []const u8,
    ty: Type,
};

pub const Enum = struct {
    name: []const u8,
    cases: []const []const u8,
};

pub const Variant = struct {
    name: []const u8,
    cases: []const Case,
};

pub const Case = struct {
    name: []const u8,
    payload: ?Type = null,
};

pub const Function = struct {
    name: []const u8,
    params: []const Param = &.{},
    result: ?Type = null,
};

pub const Param = struct {
    name: []const u8,
    ty: Type,
};

pub const Type = union(enum) {
    primitive: Primitive,
    named: []const u8,
    list: *const Type,
    option: *const Type,
    result: Result,
    tuple: []const Type,

    pub fn primitiveType(value: Primitive) Type {
        return .{ .primitive = value };
    }

    pub fn namedType(value: []const u8) Type {
        return .{ .named = value };
    }
};

pub const Primitive = enum {
    bool,
    u8,
    u16,
    u32,
    u64,
    s8,
    s16,
    s32,
    s64,
    f32,
    f64,
    string,
};

pub const Result = struct {
    ok: ?*const Type = null,
    err: ?*const Type = null,
};

pub const World = struct {
    name: []const u8,
    imports: []const WorldItem = &.{},
    exports: []const WorldItem = &.{},
};

pub const WorldItem = union(enum) {
    interface: []const u8,
    function: Function,
};

pub fn renderAlloc(allocator: std.mem.Allocator, package: Package) ![]u8 {
    try validate(package);

    var out: std.Io.Writer.Allocating = .init(allocator);
    errdefer out.deinit();

    try render(&out.writer, package);
    return out.toOwnedSlice();
}

pub fn render(writer: *std.Io.Writer, package: Package) !void {
    try writer.print("package {s};\n", .{package.id});

    for (package.interfaces) |interface| {
        try writer.writeAll("\n");
        try renderInterface(writer, interface);
    }

    for (package.worlds) |world| {
        try writer.writeAll("\n");
        try renderWorld(writer, world);
    }
}

fn renderInterface(writer: *std.Io.Writer, interface: Interface) !void {
    try writer.print("interface {s} {{\n", .{interface.name});

    for (interface.uses) |use| {
        try writer.print("  use {s}.{{", .{use.interface_name});
        for (use.names, 0..) |name, index| {
            if (index != 0) try writer.writeAll(", ");
            try writer.print("{s}", .{name});
        }
        try writer.writeAll("};\n");
    }

    if (interface.uses.len != 0 and (interface.declarations.len != 0 or interface.functions.len != 0)) {
        try writer.writeAll("\n");
    }

    for (interface.declarations, 0..) |declaration, index| {
        if (index != 0) try writer.writeAll("\n");
        try renderDeclaration(writer, declaration, "  ");
    }

    if (interface.declarations.len != 0 and interface.functions.len != 0) {
        try writer.writeAll("\n");
    }

    for (interface.functions, 0..) |function, index| {
        if (index != 0) try writer.writeAll("\n");
        try renderFunction(writer, function, "  ");
    }

    try writer.writeAll("}\n");
}

fn renderDeclaration(writer: *std.Io.Writer, declaration: Declaration, indent: []const u8) !void {
    switch (declaration) {
        .alias => |alias| {
            try writer.print("{s}type {s} = ", .{ indent, alias.name });
            try renderType(writer, alias.target);
            try writer.writeAll(";\n");
        },
        .record => |record| {
            try writer.print("{s}record {s} {{\n", .{ indent, record.name });
            for (record.fields) |field| {
                try writer.print("{s}  {s}: ", .{ indent, field.name });
                try renderType(writer, field.ty);
                try writer.writeAll(",\n");
            }
            try writer.print("{s}}}\n", .{indent});
        },
        .@"enum" => |wit_enum| {
            try writer.print("{s}enum {s} {{\n", .{ indent, wit_enum.name });
            for (wit_enum.cases) |case| {
                try writer.print("{s}  {s},\n", .{ indent, case });
            }
            try writer.print("{s}}}\n", .{indent});
        },
        .variant => |variant| {
            try writer.print("{s}variant {s} {{\n", .{ indent, variant.name });
            for (variant.cases) |case| {
                try writer.print("{s}  {s}", .{ indent, case.name });
                if (case.payload) |payload| {
                    try writer.writeAll("(");
                    try renderType(writer, payload);
                    try writer.writeAll(")");
                }
                try writer.writeAll(",\n");
            }
            try writer.print("{s}}}\n", .{indent});
        },
    }
}

fn renderFunction(writer: *std.Io.Writer, function: Function, indent: []const u8) !void {
    try writer.print("{s}{s}: func(", .{ indent, function.name });

    if (function.params.len == 0) {
        try writer.writeAll(")");
    } else {
        try writer.writeAll("\n");
        for (function.params) |param| {
            try writer.print("{s}  {s}: ", .{ indent, param.name });
            try renderType(writer, param.ty);
            try writer.writeAll(",\n");
        }
        try writer.print("{s})", .{indent});
    }

    if (function.result) |result| {
        try writer.writeAll(" -> ");
        try renderType(writer, result);
    }

    try writer.writeAll(";\n");
}

fn renderWorld(writer: *std.Io.Writer, world: World) !void {
    try writer.print("world {s} {{\n", .{world.name});

    for (world.imports) |item| {
        try renderWorldItem(writer, "import", item);
    }

    if (world.imports.len != 0 and world.exports.len != 0) {
        try writer.writeAll("\n");
    }

    for (world.exports) |item| {
        try renderWorldItem(writer, "export", item);
    }

    try writer.writeAll("}\n");
}

fn renderWorldItem(writer: *std.Io.Writer, direction: []const u8, item: WorldItem) !void {
    switch (item) {
        .interface => |name| try writer.print("  {s} {s};\n", .{ direction, name }),
        .function => |function| try renderWorldFunction(writer, direction, function),
    }
}

fn renderWorldFunction(writer: *std.Io.Writer, direction: []const u8, function: Function) !void {
    try writer.print("  {s} {s}: func(", .{ direction, function.name });

    if (function.params.len == 0) {
        try writer.writeAll(")");
    } else {
        try writer.writeAll("\n");
        for (function.params) |param| {
            try writer.print("    {s}: ", .{param.name});
            try renderType(writer, param.ty);
            try writer.writeAll(",\n");
        }
        try writer.writeAll("  )");
    }

    if (function.result) |result| {
        try writer.writeAll(" -> ");
        try renderType(writer, result);
    }

    try writer.writeAll(";\n");
}

fn renderType(writer: *std.Io.Writer, ty: Type) !void {
    switch (ty) {
        .primitive => |primitive| try writer.writeAll(primitiveName(primitive)),
        .named => |name| try writer.writeAll(name),
        .list => |child| {
            try writer.writeAll("list<");
            try renderType(writer, child.*);
            try writer.writeAll(">");
        },
        .option => |child| {
            try writer.writeAll("option<");
            try renderType(writer, child.*);
            try writer.writeAll(">");
        },
        .result => |result| {
            try writer.writeAll("result<");
            if (result.ok) |ok| {
                try renderType(writer, ok.*);
            } else {
                try writer.writeAll("_");
            }
            if (result.err) |err| {
                try writer.writeAll(", ");
                try renderType(writer, err.*);
            }
            try writer.writeAll(">");
        },
        .tuple => |items| {
            try writer.writeAll("tuple<");
            for (items, 0..) |item, index| {
                if (index != 0) try writer.writeAll(", ");
                try renderType(writer, item);
            }
            try writer.writeAll(">");
        },
    }
}

fn primitiveName(value: Primitive) []const u8 {
    return switch (value) {
        .bool => "bool",
        .u8 => "u8",
        .u16 => "u16",
        .u32 => "u32",
        .u64 => "u64",
        .s8 => "s8",
        .s16 => "s16",
        .s32 => "s32",
        .s64 => "s64",
        .f32 => "f32",
        .f64 => "f64",
        .string => "string",
    };
}

fn validate(package: Package) !void {
    try validateName(package.id, .package);

    for (package.interfaces) |interface| {
        try validateName(interface.name, .identifier);
        for (interface.uses) |use| {
            try validateName(use.interface_name, .identifier);
            for (use.names) |name| try validateName(name, .identifier);
        }
        for (interface.declarations) |declaration| try validateDeclaration(declaration);
        for (interface.functions) |function| try validateFunction(function);
    }

    for (package.worlds) |world| {
        try validateName(world.name, .identifier);
        for (world.imports) |item| try validateWorldItem(item);
        for (world.exports) |item| try validateWorldItem(item);
    }
}

fn validateDeclaration(declaration: Declaration) !void {
    switch (declaration) {
        .alias => |alias| {
            try validateName(alias.name, .identifier);
            try validateType(alias.target);
        },
        .record => |record| {
            try validateName(record.name, .identifier);
            for (record.fields) |field| {
                try validateName(field.name, .identifier);
                try validateType(field.ty);
            }
        },
        .@"enum" => |wit_enum| {
            try validateName(wit_enum.name, .identifier);
            for (wit_enum.cases) |case| try validateName(case, .identifier);
        },
        .variant => |variant| {
            try validateName(variant.name, .identifier);
            for (variant.cases) |case| {
                try validateName(case.name, .identifier);
                if (case.payload) |payload| try validateType(payload);
            }
        },
    }
}

fn validateFunction(function: Function) !void {
    try validateName(function.name, .identifier);
    for (function.params) |param| {
        try validateName(param.name, .identifier);
        try validateType(param.ty);
    }
    if (function.result) |result| try validateType(result);
}

fn validateWorldItem(item: WorldItem) !void {
    switch (item) {
        .interface => |name| try validateName(name, .interface_ref),
        .function => |function| try validateFunction(function),
    }
}

fn validateType(ty: Type) !void {
    switch (ty) {
        .primitive => {},
        .named => |name| try validateName(name, .type_ref),
        .list => |child| try validateType(child.*),
        .option => |child| try validateType(child.*),
        .result => |result| {
            if (result.ok) |ok| try validateType(ok.*);
            if (result.err) |err| try validateType(err.*);
        },
        .tuple => |items| for (items) |item| try validateType(item),
    }
}

const NameKind = enum {
    package,
    identifier,
    interface_ref,
    type_ref,
};

fn validateName(name: []const u8, kind: NameKind) !void {
    if (name.len == 0) return error.EmptyWitName;

    for (name) |char| {
        const ok = switch (kind) {
            .package => std.ascii.isAlphanumeric(char) or char == ':' or char == '-' or char == '.' or char == '@',
            .identifier => std.ascii.isAlphanumeric(char) or char == '-',
            .interface_ref, .type_ref => std.ascii.isAlphanumeric(char) or char == '-' or char == ':' or char == '/' or char == '.',
        };
        if (!ok) return error.InvalidWitName;
    }
}

test "renders a compact package" {
    const allocator = std.testing.allocator;
    const u32_type = Type.primitiveType(.u32);
    const string_type = Type.primitiveType(.string);
    const error_type = Type.namedType("call-error");
    const result_type = Type{ .result = .{ .ok = &u32_type, .err = &error_type } };

    const package = Package{
        .id = "local:demo@1.0.0",
        .interfaces = &.{.{
            .name = "api",
            .declarations = &.{
                .{ .@"enum" = .{ .name = "call-error", .cases = &.{ "bad-input", "busy" } } },
            },
            .functions = &.{.{
                .name = "call",
                .params = &.{.{ .name = "name", .ty = string_type }},
                .result = result_type,
            }},
        }},
        .worlds = &.{.{
            .name = "guest",
            .imports = &.{.{ .interface = "api" }},
        }},
    };

    const rendered = try renderAlloc(allocator, package);
    defer allocator.free(rendered);

    try std.testing.expectEqualStrings(
        \\package local:demo@1.0.0;
        \\
        \\interface api {
        \\  enum call-error {
        \\    bad-input,
        \\    busy,
        \\  }
        \\
        \\  call: func(
        \\    name: string,
        \\  ) -> result<u32, call-error>;
        \\}
        \\
        \\world guest {
        \\  import api;
        \\}
        \\
    , rendered);
}

test "rejects invalid names" {
    try std.testing.expectError(error.InvalidWitName, validate(Package{
        .id = "local:bad name",
    }));
}
