//! Executable, deliberately bounded Component Model profile: list<u8, 4> -> u32,
//! list<u8> -> u32 and list<u8> -> list<u8>. Decode all declarations before
//! allowing any guest code to run. References are resolved in declaration order
//! in their respective index spaces, not by layout.
const std = @import("std");
const module = @import("module.zig");
const validator = @import("validator.zig");

pub const profile = "fixed-list-u8-u32/1";
pub const max_component_bytes = 1024 * 1024;
const max_items = 256;

// wat 0.259.0 output from Plexus fixtures/fixed-length-lists/component/checksum.wat.
const reference_component =
    "\x00\x61\x73\x6d\x0d\x00\x01\x00\x01\x48\x00\x61\x73\x6d\x01\x00\x00\x00\x01\x09\x01\x60\x04\x7f\x7f\x7f\x7f\x01\x7f\x03\x02\x01" ++
    "\x00\x07\x0c\x01\x08\x63\x68\x65\x63\x6b\x73\x75\x6d\x00\x00\x0a\x0f\x01\x0d\x00\x20\x00\x20\x01\x6a\x20\x02\x6a\x20\x03\x6a\x0b" ++
    "\x00\x10\x04\x6e\x61\x6d\x65\x00\x09\x08\x63\x68\x65\x63\x6b\x73\x75\x6d\x02\x04\x01\x00\x00\x00\x07\x0f\x02\x67\x7d\x04\x40\x01" ++
    "\x05\x76\x61\x6c\x75\x65\x00\x00\x79\x06\x0e\x01\x00\x00\x01\x00\x08\x63\x68\x65\x63\x6b\x73\x75\x6d\x08\x06\x01\x00\x00\x00\x00" ++
    "\x01\x0b\x0e\x01\x00\x08\x63\x68\x65\x63\x6b\x73\x75\x6d\x01\x00\x00\x00\x4d\x0e\x63\x6f\x6d\x70\x6f\x6e\x65\x6e\x74\x2d\x6e\x61" ++
    "\x6d\x65\x01\x0d\x00\x11\x01\x00\x08\x63\x68\x65\x63\x6b\x73\x75\x6d\x01\x13\x00\x12\x01\x00\x0e\x69\x6d\x70\x6c\x65\x6d\x65\x6e" ++
    "\x74\x61\x74\x69\x6f\x6e\x01\x18\x03\x02\x00\x05\x62\x79\x74\x65\x73\x01\x0d\x63\x68\x65\x63\x6b\x73\x75\x6d\x2d\x74\x79\x70\x65";

// wat 0.259.0 output from Plexus fixtures/fixed-length-lists/component/checksum-list.wat.
// Its canon lift declares (memory $memory) and (realloc $realloc).
const list_component =
    "\x00\x61\x73\x6d\x0d\x00\x01\x00\x01\x8e\x02\x00\x61\x73\x6d\x01\x00\x00\x00\x01\x0f\x02\x60\x04\x7f\x7f\x7f\x7f\x01\x7f\x60\x02" ++
    "\x7f\x7f\x01\x7f\x03\x03\x02\x00\x01\x05\x03\x01\x00\x01\x06\x06\x01\x7f\x01\x41\x08\x0b\x07\x24\x03\x06\x6d\x65\x6d\x6f\x72\x79" ++
    "\x02\x00\x0c\x63\x61\x62\x69\x5f\x72\x65\x61\x6c\x6c\x6f\x63\x00\x00\x08\x63\x68\x65\x63\x6b\x73\x75\x6d\x00\x01\x0a\x4c\x02\x20" ++
    "\x01\x01\x7f\x23\x00\x20\x02\x41\x01\x6b\x6a\x20\x02\x41\x01\x6b\x41\x7f\x73\x71\x21\x04\x20\x04\x20\x03\x6a\x24\x00\x20\x04\x0b" ++
    "\x29\x01\x02\x7f\x02\x40\x03\x40\x20\x03\x20\x01\x4f\x0d\x01\x20\x02\x20\x00\x20\x03\x6a\x2d\x00\x00\x6a\x21\x02\x20\x03\x41\x01" ++
    "\x6a\x21\x03\x0c\x00\x0b\x0b\x20\x02\x0b\x00\x6d\x04\x6e\x61\x6d\x65\x00\x09\x08\x63\x68\x65\x63\x6b\x73\x75\x6d\x02\x3c\x02\x00" ++
    "\x05\x00\x03\x6f\x6c\x64\x01\x08\x6f\x6c\x64\x5f\x73\x69\x7a\x65\x02\x05\x61\x6c\x69\x67\x6e\x03\x04\x73\x69\x7a\x65\x04\x03\x70" ++
    "\x74\x72\x01\x04\x00\x03\x70\x74\x72\x01\x03\x6c\x65\x6e\x02\x03\x73\x75\x6d\x03\x05\x69\x6e\x64\x65\x78\x03\x14\x01\x01\x02\x00" ++
    "\x04\x64\x6f\x6e\x65\x01\x09\x6e\x65\x78\x74\x2d\x62\x79\x74\x65\x07\x07\x01\x00\x04\x6e\x65\x78\x74\x02\x04\x01\x00\x00\x00\x06" ++
    "\x1d\x02\x00\x02\x01\x00\x06\x6d\x65\x6d\x6f\x72\x79\x00\x00\x01\x00\x0c\x63\x61\x62\x69\x5f\x72\x65\x61\x6c\x6c\x6f\x63\x07\x0e" ++
    "\x02\x70\x7d\x40\x01\x05\x62\x79\x74\x65\x73\x00\x00\x79\x06\x0e\x01\x00\x00\x01\x00\x08\x63\x68\x65\x63\x6b\x73\x75\x6d\x08\x0a" ++
    "\x01\x00\x00\x01\x02\x03\x00\x04\x00\x01\x0b\x0e\x01\x00\x08\x63\x68\x65\x63\x6b\x73\x75\x6d\x01\x00\x00\x00\x68\x0e\x63\x6f\x6d" ++
    "\x70\x6f\x6e\x65\x6e\x74\x2d\x6e\x61\x6d\x65\x01\x0c\x00\x00\x01\x00\x07\x72\x65\x61\x6c\x6c\x6f\x63\x01\x0b\x00\x02\x01\x00\x06" ++
    "\x6d\x65\x6d\x6f\x72\x79\x01\x0d\x00\x11\x01\x00\x08\x63\x68\x65\x63\x6b\x73\x75\x6d\x01\x13\x00\x12\x01\x00\x0e\x69\x6d\x70\x6c" ++
    "\x65\x6d\x65\x6e\x74\x61\x74\x69\x6f\x6e\x01\x18\x03\x02\x00\x05\x62\x79\x74\x65\x73\x01\x0d\x63\x68\x65\x63\x6b\x73\x75\x6d\x2d" ++
    "\x74\x79\x70\x65";

/// How the canonical ABI carries the argument list of the lifted function.
pub const Lowering = union(enum) {
    /// A fixed-length list flattens into one i32 core parameter per element.
    flattened: u32,
    /// A list of unknown length is written into the component's own memory by
    /// its own allocator, and passed as the pointer and length pair.
    memory: Allocation,
};

pub const Allocation = struct {
    realloc_export: []const u8,
    realloc_index: u32,
};

// wat 0.259.0 output from Plexus fixtures/fixed-length-lists/component/scan.wat.
// Its lift declares a list result, with memory, realloc and post-return.
const scan_component =
    "\x00\x61\x73\x6d\x0d\x00\x01\x00\x01\x87\x03\x00\x61\x73\x6d\x01\x00\x00\x00\x01\x13\x03\x60\x04\x7f\x7f\x7f\x7f\x01\x7f\x60\x02" ++
    "\x7f\x7f\x01\x7f\x60\x01\x7f\x00\x03\x04\x03\x00\x01\x02\x05\x03\x01\x00\x01\x06\x0b\x02\x7f\x01\x41\x08\x0b\x7f\x01\x41\x00\x0b" ++
    "\x07\x31\x04\x06\x6d\x65\x6d\x6f\x72\x79\x02\x00\x0c\x63\x61\x62\x69\x5f\x72\x65\x61\x6c\x6c\x6f\x63\x00\x00\x04\x73\x63\x61\x6e" ++
    "\x00\x01\x0e\x63\x61\x62\x69\x5f\x70\x6f\x73\x74\x5f\x73\x63\x61\x6e\x00\x02\x0a\x86\x01\x03\x20\x01\x01\x7f\x23\x00\x20\x02\x41" ++
    "\x01\x6b\x6a\x20\x02\x41\x01\x6b\x41\x7f\x73\x71\x21\x04\x20\x04\x20\x03\x6a\x24\x00\x20\x04\x0b\x59\x01\x04\x7f\x41\x00\x41\x00" ++
    "\x41\x01\x20\x01\x10\x00\x21\x02\x02\x40\x03\x40\x20\x04\x20\x01\x4f\x0d\x01\x20\x05\x20\x00\x20\x04\x6a\x2d\x00\x00\x6a\x21\x05" ++
    "\x20\x02\x20\x04\x6a\x20\x05\x3a\x00\x00\x20\x04\x41\x01\x6a\x21\x04\x0c\x00\x0b\x0b\x41\x00\x41\x00\x41\x04\x41\x08\x10\x00\x21" ++
    "\x03\x20\x03\x20\x02\x36\x02\x00\x20\x03\x20\x01\x36\x02\x04\x20\x03\x0b\x09\x00\x23\x01\x41\x01\x6a\x24\x01\x0b\x00\x93\x01\x04" ++
    "\x6e\x61\x6d\x65\x00\x05\x04\x73\x63\x61\x6e\x01\x0b\x01\x00\x08\x61\x6c\x6c\x6f\x63\x61\x74\x65\x02\x4f\x03\x00\x05\x00\x03\x6f" ++
    "\x6c\x64\x01\x08\x6f\x6c\x64\x5f\x73\x69\x7a\x65\x02\x05\x61\x6c\x69\x67\x6e\x03\x04\x73\x69\x7a\x65\x04\x03\x70\x74\x72\x01\x06" ++
    "\x00\x03\x70\x74\x72\x01\x03\x6c\x65\x6e\x02\x03\x6f\x75\x74\x03\x04\x61\x72\x65\x61\x04\x05\x69\x6e\x64\x65\x78\x05\x03\x73\x75" ++
    "\x6d\x02\x01\x00\x04\x61\x72\x65\x61\x03\x14\x01\x01\x02\x00\x04\x64\x6f\x6e\x65\x01\x09\x6e\x65\x78\x74\x2d\x62\x79\x74\x65\x07" ++
    "\x11\x02\x00\x04\x6e\x65\x78\x74\x01\x08\x72\x65\x6c\x65\x61\x73\x65\x64\x02\x04\x01\x00\x00\x00\x06\x30\x03\x00\x02\x01\x00\x06" ++
    "\x6d\x65\x6d\x6f\x72\x79\x00\x00\x01\x00\x0c\x63\x61\x62\x69\x5f\x72\x65\x61\x6c\x6c\x6f\x63\x00\x00\x01\x00\x0e\x63\x61\x62\x69" ++
    "\x5f\x70\x6f\x73\x74\x5f\x73\x63\x61\x6e\x07\x0e\x02\x70\x7d\x40\x01\x05\x62\x79\x74\x65\x73\x00\x00\x00\x06\x0a\x01\x00\x00\x01" ++
    "\x00\x04\x73\x63\x61\x6e\x08\x0c\x01\x00\x00\x02\x03\x03\x00\x04\x00\x05\x01\x01\x0b\x0a\x01\x00\x04\x73\x63\x61\x6e\x01\x00\x00" ++
    "\x00\x6d\x0e\x63\x6f\x6d\x70\x6f\x6e\x65\x6e\x74\x2d\x6e\x61\x6d\x65\x01\x19\x00\x00\x02\x00\x07\x72\x65\x61\x6c\x6c\x6f\x63\x01" ++
    "\x0b\x70\x6f\x73\x74\x2d\x72\x65\x74\x75\x72\x6e\x01\x0b\x00\x02\x01\x00\x06\x6d\x65\x6d\x6f\x72\x79\x01\x09\x00\x11\x01\x00\x04" ++
    "\x73\x63\x61\x6e\x01\x13\x00\x12\x01\x00\x0e\x69\x6d\x70\x6c\x65\x6d\x65\x6e\x74\x61\x74\x69\x6f\x6e\x01\x14\x03\x02\x00\x05\x62" ++
    "\x79\x74\x65\x73\x01\x09\x73\x63\x61\x6e\x2d\x74\x79\x70\x65";

/// How the canonical ABI carries the result back out of the lifted function.
pub const Lifting = union(enum) {
    /// A u32 fits in the one flattened core result.
    scalar,
    /// A list does not, so the callee returns a pointer to its own return area
    /// holding the pointer and length, and post-return releases it afterwards.
    memory: ReturnArea,
};

pub const ReturnArea = struct {
    post_return_export: ?[]const u8,
};

/// Slices borrow the input component. No allocation survives resolve().
pub const Resolved = struct {
    module_bytes: []const u8,
    core_export: []const u8,
    core_function_index: u32,
    component_function_index: u32,
    component_type_index: u32,
    lowering: Lowering,
    lifting: Lifting,
};

/// Whether a component function takes a fixed-length list or one of any length.
const Parameter = enum { fixed_bytes, list_bytes };

/// Whether it answers with a scalar or with a list.
const ResultType = enum { u32_result, list_result };

const Signature = struct {
    name: []const u8, // Parameter name is part of component function typing.
    parameter: Parameter,
    result: ResultType,
};

const Type = union(enum) {
    u8_type,
    u32_type,
    fixed_bytes,
    list_bytes,
    function: Signature,
};
const CoreFunction = struct { name: []const u8, index: u32 };
const Function = struct { core: CoreFunction, type_index: u32, lowering: Lowering, lifting: Lifting };

pub fn resolve(allocator: std.mem.Allocator, bytes: []const u8, export_name: []const u8) !Resolved {
    if (bytes.len > max_component_bytes) return error.ComponentTooLarge;
    if (bytes.len < 8 or !std.mem.eql(u8, bytes[0..8], "\x00asm\x0d\x00\x01\x00"))
        return error.InvalidComponentHeader;
    var reader = Reader{ .bytes = bytes, .offset = 8 };
    var parsed: ?module.Module = null;
    defer if (parsed) |*m| m.deinit(allocator);
    var instantiated = false;
    var types: std.ArrayList(Type) = .empty;
    defer types.deinit(allocator);
    var core_functions: std.ArrayList(CoreFunction) = .empty;
    defer core_functions.deinit(allocator);
    // Aliased core memories, in declaration order: canonical options index them.
    var core_memories: std.ArrayList(u32) = .empty;
    defer core_memories.deinit(allocator);
    var functions: std.ArrayList(Function) = .empty;
    defer functions.deinit(allocator);
    var export_names: std.ArrayList([]const u8) = .empty;
    defer export_names.deinit(allocator);
    var result: ?Resolved = null;
    var sections: usize = 0;
    while (!reader.atEnd()) {
        sections += 1;
        if (sections > 1024) return error.ComponentTooComplex;
        const id = try reader.byte();
        var section = Reader{ .bytes = try reader.take(try reader.u32leb()) };
        switch (id) {
            0 => { // Custom payload after its UTF-8 name is intentionally opaque.
                _ = try section.name();
                continue;
            },
            1 => {
                if (parsed != null) return error.UnsupportedMultipleCoreModules;
                try preflightCore(section.bytes);
                parsed = try module.Module.parse(allocator, section.bytes);
                if (parsed.?.imports.items.len != 0) return error.ComponentImportsForbidden;
                var parameter_total: usize = 0;
                for (parsed.?.function_types.items) |typ| {
                    if (typ.params.len > max_items) return error.ComponentTooComplex;
                }
                for (parsed.?.functions.items) |func| {
                    parameter_total += (try parsed.?.functionType(func.type_index)).params.len;
                    if (parameter_total > 65536) return error.ComponentTooComplex;
                }
                try validator.validate(allocator, &parsed.?);
                try uniqueCoreExports(parsed.?);
                continue;
            },
            2 => {
                const count = try section.count();
                for (0..count) |_| {
                    if (instantiated) return error.UnsupportedMultipleCoreInstances;
                    if (try section.byte() != 0) return error.UnsupportedCoreInstance;
                    if (try section.u32leb() != 0 or parsed == null) return error.InvalidComponentModuleIndex;
                    if (try section.u32leb() != 0) return error.ComponentImportsForbidden;
                    instantiated = true;
                }
            },
            6 => {
                const count = try section.count();
                for (0..count) |_| {
                    // alias core instance export, of the core function or memory sort.
                    if (try section.byte() != 0) return error.UnsupportedComponentAlias;
                    const sort = try section.byte();
                    if (sort != 0 and sort != 2) return error.UnsupportedComponentAlias;
                    if (try section.byte() != 1) return error.UnsupportedComponentAlias;
                    if (try section.u32leb() != 0 or !instantiated) return error.InvalidComponentInstanceIndex;
                    const name = try section.name();
                    if (sort == 0) {
                        const core = try findCoreFunction(parsed.?, name);
                        try appendBounded(CoreFunction, allocator, &core_functions, core);
                    } else {
                        try appendBounded(u32, allocator, &core_memories, try findCoreMemory(parsed.?, name));
                    }
                }
            },
            7 => {
                const count = try section.count();
                for (0..count) |_| {
                    const typ: Type = switch (try section.byte()) {
                        0x7d => .u8_type,
                        0x79 => .u32_type,
                        0x67 => blk: {
                            if (try valueType(&section, types.items) != .u8_type or try section.u32leb() != 4)
                                return error.UnsupportedComponentType;
                            break :blk .fixed_bytes;
                        },
                        0x70 => blk: {
                            if (try valueType(&section, types.items) != .u8_type)
                                return error.UnsupportedComponentType;
                            break :blk .list_bytes;
                        },
                        0x40 => blk: {
                            if (try section.u32leb() != 1) return error.UnsupportedComponentSignature;
                            const name = try section.name();
                            try simpleName(name);
                            const parameter: Parameter = switch (try valueType(&section, types.items)) {
                                .fixed_bytes => .fixed_bytes,
                                .list_bytes => .list_bytes,
                                else => return error.UnsupportedComponentSignature,
                            };
                            if (try section.byte() != 0) return error.UnsupportedComponentSignature;
                            const answer: ResultType = switch (try valueType(&section, types.items)) {
                                .u32_type => .u32_result,
                                .list_bytes => .list_result,
                                else => return error.UnsupportedComponentSignature,
                            };
                            break :blk .{ .function = .{ .name = name, .parameter = parameter, .result = answer } };
                        },
                        else => return error.UnsupportedComponentType,
                    };
                    try appendBounded(Type, allocator, &types, typ);
                }
            },
            8 => {
                const count = try section.count();
                for (0..count) |_| {
                    if (try section.byte() != 0 or try section.byte() != 0) return error.UnsupportedCanonicalOperation;
                    const core_index = try section.u32leb();
                    if (core_index >= core_functions.items.len) return error.InvalidCanonicalFunctionIndex;
                    const options = try canonicalOptions(&section, core_functions.items, core_memories.items);
                    const type_index = try section.u32leb();
                    try functionType(types.items, type_index);
                    const core = core_functions.items[core_index];
                    const signature = types.items[type_index].function;
                    const lowered = try lowering(signature.parameter, options, parsed.?);
                    const lifted = try lifting(signature.result, options, parsed.?);
                    try checkCoreSignature(parsed.?, core.index, lowered);
                    try appendBounded(Function, allocator, &functions, .{ .core = core, .type_index = type_index, .lowering = lowered, .lifting = lifted });
                }
            },
            11 => {
                const count = try section.count();
                for (0..count) |_| {
                    // The 0/1 discriminators are the current/legacy plain-name encodings.
                    if (try section.byte() > 1) return error.UnsupportedComponentExportName;
                    const name = try section.name();
                    try simpleName(name);
                    for (export_names.items) |existing| {
                        if (std.mem.eql(u8, name, existing)) return error.DuplicateComponentExport;
                    }
                    if (try section.byte() != 1) return error.UnsupportedComponentExport;
                    const index = try section.u32leb();
                    if (index >= functions.items.len) return error.InvalidComponentFunctionIndex;
                    const func = functions.items[index];
                    switch (try section.byte()) {
                        0 => {},
                        1 => {
                            if (try section.byte() != 1) return error.UnsupportedComponentExportType;
                            const type_index = try section.u32leb();
                            try functionType(types.items, type_index);
                            const declared = types.items[type_index].function;
                            const actual = types.items[func.type_index].function;
                            if (!std.mem.eql(u8, declared.name, actual.name) or
                                declared.parameter != actual.parameter or declared.result != actual.result)
                                return error.ComponentExportTypeMismatch;
                        },
                        else => return error.InvalidComponentExportType,
                    }
                    if (std.mem.eql(u8, name, export_name)) result = .{
                        .module_bytes = parsed.?.bytes,
                        .core_export = func.core.name,
                        .core_function_index = func.core.index,
                        .component_function_index = index,
                        .component_type_index = func.type_index,
                        .lowering = func.lowering,
                        .lifting = func.lifting,
                    };
                    try appendBounded([]const u8, allocator, &export_names, name);
                    // Component exports introduce another item in the corresponding
                    // index space. Later declarations can reference this alias.
                    try appendBounded(Function, allocator, &functions, func);
                }
            },
            10 => return error.ComponentImportsForbidden,
            else => return error.UnsupportedComponentSection,
        }
        if (!section.atEnd()) return error.TrailingComponentSectionBytes;
    }
    if (!instantiated) return error.MissingComponentInstance;
    return result orelse error.MissingComponentExport;
}

fn appendBounded(comptime T: type, allocator: std.mem.Allocator, list: *std.ArrayList(T), item: T) !void {
    if (list.items.len >= max_items) return error.ComponentTooComplex;
    try list.append(allocator, item);
}

fn functionType(types: []const Type, index: u32) !void {
    if (index >= types.len) return error.InvalidComponentTypeIndex;
    if (types[index] != .function) return error.UnsupportedComponentSignature;
}

fn valueType(reader: *Reader, types: []const Type) !Type {
    if (reader.offset >= reader.bytes.len) return error.UnexpectedEndOfComponent;
    switch (reader.bytes[reader.offset]) {
        0x7d => {
            reader.offset += 1;
            return .u8_type;
        },
        0x79 => {
            reader.offset += 1;
            return .u32_type;
        },
        else => {},
    }
    const index = try reader.u32leb();
    // Type indices use s33, so a terminating sign bit denotes a negative value.
    if ((reader.bytes[reader.offset - 1] & 0x40) != 0) return error.UnsupportedComponentType;
    if (index >= types.len) return error.InvalidComponentTypeIndex;
    if (types[index] == .function) return error.UnsupportedComponentType;
    return types[index];
}

fn simpleName(name: []const u8) !void {
    // This first profile supports unqualified lowercase kebab names only.
    var at_start = true;
    for (name) |char| {
        if (char >= 'a' and char <= 'z') {
            at_start = false;
            continue;
        }
        if (!at_start and char >= '0' and char <= '9') continue;
        if (!at_start and char == '-') {
            at_start = true;
            continue;
        }
        return error.UnsupportedComponentName;
    }
    if (at_start) return error.UnsupportedComponentName;
}

fn findCoreMemory(parsed: module.Module, name: []const u8) !u32 {
    for (parsed.exports.items) |exp| {
        if (std.mem.eql(u8, exp.name, name)) {
            if (exp.kind != .memory) return error.ComponentAliasKindMismatch;
            return exp.index;
        }
    }
    return error.MissingCoreExport;
}

fn findCoreFunction(parsed: module.Module, name: []const u8) !CoreFunction {
    for (parsed.exports.items) |exp| {
        if (std.mem.eql(u8, exp.name, name)) {
            if (exp.kind != .function) return error.ComponentAliasKindMismatch;
            return .{ .name = exp.name, .index = exp.index };
        }
    }
    return error.MissingCoreExport;
}

const Options = struct {
    memory: ?u32 = null,
    realloc: ?CoreFunction = null,
    post_return: ?CoreFunction = null,
};

/// Canonical options, in the binary encoding. Only what a list of unknown
/// length needs is accepted: its bytes live in the component's own memory.
fn canonicalOptions(
    section: *Reader,
    core_functions: []const CoreFunction,
    core_memories: []const u32,
) !Options {
    var options = Options{};
    const count = try section.u32leb();
    if (count > 3) return error.UnsupportedCanonicalOptions;
    for (0..count) |_| {
        switch (try section.byte()) {
            0x03 => {
                if (options.memory != null) return error.DuplicateCanonicalOption;
                const index = try section.u32leb();
                if (index >= core_memories.len) return error.InvalidCanonicalMemoryIndex;
                options.memory = core_memories[index];
            },
            0x04 => {
                if (options.realloc != null) return error.DuplicateCanonicalOption;
                const index = try section.u32leb();
                if (index >= core_functions.len) return error.InvalidCanonicalFunctionIndex;
                options.realloc = core_functions[index];
            },
            0x05 => {
                if (options.post_return != null) return error.DuplicateCanonicalOption;
                const index = try section.u32leb();
                if (index >= core_functions.len) return error.InvalidCanonicalFunctionIndex;
                options.post_return = core_functions[index];
            },
            else => return error.UnsupportedCanonicalOptions,
        }
    }
    return options;
}

/// Match the declared parameter against the options that can carry it.
fn lowering(parameter: Parameter, options: Options, parsed: module.Module) !Lowering {
    switch (parameter) {
        .fixed_bytes => {
            // Four flattened i32 parameters need no memory and no allocator.
            if (options.realloc != null) return error.UnsupportedCanonicalOptions;
            return .{ .flattened = 4 };
        },
        .list_bytes => {
            const realloc = options.realloc orelse return error.MissingCanonicalRealloc;
            try checkMemoryOption(options);
            try checkReallocSignature(parsed, realloc.index);
            return .{ .memory = .{ .realloc_export = realloc.name, .realloc_index = realloc.index } };
        },
    }
}

/// Match the declared result against the options that can carry it back.
fn lifting(result: ResultType, options: Options, parsed: module.Module) !Lifting {
    switch (result) {
        .u32_result => {
            // One flattened core result needs no return area to release.
            if (options.post_return != null) return error.UnsupportedCanonicalOptions;
            return .scalar;
        },
        .list_result => {
            try checkMemoryOption(options);
            if (options.post_return) |post_return| {
                try checkPostReturnSignature(parsed, post_return.index);
                return .{ .memory = .{ .post_return_export = post_return.name } };
            }
            return .{ .memory = .{ .post_return_export = null } };
        },
    }
}

fn checkMemoryOption(options: Options) !void {
    const memory = options.memory orelse return error.MissingCanonicalMemory;
    // This profile runs one core instance, whose only memory is index 0.
    if (memory != 0) return error.UnsupportedCanonicalMemoryIndex;
}

/// The lowering fixes the core parameters; the lifting keeps one core result,
/// which is either the u32 itself or the pointer to the callee's return area.
fn checkCoreSignature(parsed: module.Module, index: u32, lowered: Lowering) !void {
    if (index >= parsed.functions.items.len) return error.InvalidCoreFunctionIndex;
    const typ = try parsed.functionType(parsed.functions.items[index].type_index);
    const expected: []const module.ValueType = switch (lowered) {
        // Element per parameter, against pointer and length.
        .flattened => &.{ .i32, .i32, .i32, .i32 },
        .memory => &.{ .i32, .i32 },
    };
    if (!std.mem.eql(module.ValueType, typ.params, expected) or
        !std.mem.eql(module.ValueType, typ.results, &.{.i32})) return error.CanonicalSignatureMismatch;
}

fn checkPostReturnSignature(parsed: module.Module, index: u32) !void {
    if (index >= parsed.functions.items.len) return error.InvalidCoreFunctionIndex;
    const typ = try parsed.functionType(parsed.functions.items[index].type_index);
    // Post-return receives the core results it is releasing, and answers nothing.
    if (!std.mem.eql(module.ValueType, typ.params, &.{.i32}) or typ.results.len != 0)
        return error.PostReturnSignatureMismatch;
}

fn checkReallocSignature(parsed: module.Module, index: u32) !void {
    if (index >= parsed.functions.items.len) return error.InvalidCoreFunctionIndex;
    const typ = try parsed.functionType(parsed.functions.items[index].type_index);
    // (original, original size, alignment, new size) -> pointer
    if (!std.mem.eql(module.ValueType, typ.params, &.{ .i32, .i32, .i32, .i32 }) or
        !std.mem.eql(module.ValueType, typ.results, &.{.i32})) return error.ReallocSignatureMismatch;
}

fn uniqueCoreExports(parsed: module.Module) !void {
    for (parsed.exports.items, 0..) |exp, i| {
        if (!std.unicode.utf8ValidateSlice(exp.name)) return error.InvalidComponentUtf8;
        for (parsed.exports.items[0..i]) |prior| {
            if (std.mem.eql(u8, exp.name, prior.name)) return error.DuplicateCoreExport;
        }
    }
}

/// The general core parser predates strict section ordering and bounded local
/// counts. Establish those invariants before calling it or allocating locals.
fn preflightCore(bytes: []const u8) !void {
    if (bytes.len < 8 or !std.mem.eql(u8, bytes[0..8], "\x00asm\x01\x00\x00\x00")) return error.InvalidCoreHeader;
    var reader = Reader{ .bytes = bytes, .offset = 8 };
    var previous_rank: u8 = 0;
    var total_locals: usize = 0;
    var section_count: usize = 0;
    while (!reader.atEnd()) {
        section_count += 1;
        if (section_count > 1024) return error.ComponentTooComplex;
        const id = try reader.byte();
        var section = Reader{ .bytes = try reader.take(try reader.u32leb()) };
        if (id == 0) {
            _ = try section.name();
            continue;
        }
        const rank: u8 = switch (id) {
            1...9 => id,
            12 => 10,
            10 => 11,
            11 => 12,
            else => return error.UnsupportedCoreSection,
        };
        if (rank <= previous_rank) return error.InvalidCoreSectionOrder;
        previous_rank = rank;
        // Tables/element segments are outside this profile, including the large
        // index-vector allocations they could otherwise request during parsing.
        if (id == 4 or id == 9) return error.UnsupportedComponentCoreTables;
        if (id != 8 and id != 12) {
            var vector = section;
            const count = try vector.u32leb();
            if (count > 8192 or (id == 7 and count > max_items)) return error.ComponentTooComplex;
        }
        if (id == 2) {
            if (try section.u32leb() != 0) return error.ComponentImportsForbidden;
            if (!section.atEnd()) return error.TrailingCoreSectionBytes;
        }
        if (id == 10) {
            const count = try section.u32leb();
            if (count > 8192) return error.ComponentTooComplex;
            for (0..count) |_| {
                var body = Reader{ .bytes = try section.take(try section.u32leb()) };
                const groups = try body.u32leb();
                if (groups > 8192) return error.ComponentTooComplex;
                for (0..groups) |_| {
                    const locals = try body.u32leb();
                    if (locals > 8192 or total_locals + locals > 65536) return error.ComponentTooComplex;
                    total_locals += locals;
                    _ = try body.byte();
                }
            }
            if (!section.atEnd()) return error.TrailingCoreSectionBytes;
        }
    }
}

const Reader = struct {
    bytes: []const u8,
    offset: usize = 0,
    fn atEnd(self: Reader) bool {
        return self.offset == self.bytes.len;
    }
    fn byte(self: *Reader) !u8 {
        if (self.offset >= self.bytes.len) return error.UnexpectedEndOfComponent;
        defer self.offset += 1;
        return self.bytes[self.offset];
    }
    fn take(self: *Reader, len: u32) ![]const u8 {
        if (len > self.bytes.len - self.offset) return error.UnexpectedEndOfComponent;
        const start = self.offset;
        self.offset += len;
        return self.bytes[start..self.offset];
    }
    fn u32leb(self: *Reader) !u32 {
        var result: u32 = 0;
        for (0..5) |i| {
            const b = try self.byte();
            if (i == 4 and b > 0x0f) return error.InvalidComponentLeb;
            result |= @as(u32, b & 0x7f) << @as(u5, @intCast(i * 7));
            if (b & 0x80 == 0) return result;
        }
        return error.InvalidComponentLeb;
    }
    fn count(self: *Reader) !u32 {
        const n = try self.u32leb();
        if (n > max_items) return error.ComponentTooComplex;
        return n;
    }
    fn name(self: *Reader) ![]const u8 {
        const n = try self.u32leb();
        if (n > 1024) return error.ComponentNameTooLong;
        const bytes = try self.take(n);
        if (!std.unicode.utf8ValidateSlice(bytes)) return error.InvalidComponentUtf8;
        return bytes;
    }
};

test "real reference component resolves its declared canonical ABI" {
    const resolved = try resolve(std.testing.allocator, reference_component, "checksum");
    try std.testing.expectEqualStrings("checksum", resolved.core_export);
    try std.testing.expectEqual(@as(u32, 0), resolved.core_function_index);
    try std.testing.expectEqual(@as(u32, 1), resolved.component_type_index);
    try std.testing.expectEqualSlices(u8, reference_component[10..82], resolved.module_bytes);
    try std.testing.expectError(error.MissingComponentExport, resolve(std.testing.allocator, reference_component, "absent"));
}

test "component reference resolves renamed exports without relying on code bytes" {
    var changed = reference_component.*;
    // Three semantic occurrences: core export, core alias and component export.
    for ([_]usize{ 37, 113, 134 }) |offset| @memcpy(changed[offset..][0..8], "other-fn");
    // The guest computes xor rather than add; name custom sections remain old.
    for ([_]usize{ 56, 59, 62 }) |offset| changed[offset] = 0x73;
    const resolved = try resolve(std.testing.allocator, &changed, "other-fn");
    try std.testing.expectEqualStrings("other-fn", resolved.core_export);
}

test "component index spaces include types aliases canonical functions and exports" {
    // Repeat a valid type declaration, alias and lift; then reference nonzero
    // indices. Export 'first' itself adds a new component function at index two.
    const bytes = reference_component[0..88] ++
        "\x07\x10\x03\x7d\x67\x00\x04\x40\x01\x05value\x01\x00\x79" ++
        reference_component[105..121] ++ reference_component[105..121] ++
        "\x08\x0b\x02\x00\x00\x00\x00\x02\x00\x00\x01\x00\x02" ++
        "\x0b\x16\x02\x00\x05first\x01\x01\x00\x00\x06second\x01\x02\x00";
    const resolved = try resolve(std.testing.allocator, bytes, "second");
    try std.testing.expectEqual(@as(u32, 2), resolved.component_function_index);
    try std.testing.expectEqual(@as(u32, 2), resolved.component_type_index);
}

test "component resolver checks every declaration including trailing ones" {
    try std.testing.expectError(error.UnsupportedComponentType, resolve(std.testing.allocator, reference_component ++ "\x07\x02\x01\x73", "checksum"));
    try std.testing.expectError(error.UnsupportedComponentSection, resolve(std.testing.allocator, reference_component ++ "\x09\x01\x00", "checksum"));
    try std.testing.expectError(error.ComponentImportsForbidden, resolve(std.testing.allocator, reference_component ++ "\x0a\x01\x00", "checksum"));
    try std.testing.expectError(error.TrailingComponentSectionBytes, resolve(std.testing.allocator, reference_component ++ "\x07\x02\x00\x00", "checksum"));
    try std.testing.expectError(error.DuplicateComponentExport, resolve(std.testing.allocator, reference_component ++ reference_component[129..145], "checksum"));
    try std.testing.expectError(error.InvalidComponentFunctionIndex, resolve(std.testing.allocator, reference_component ++ "\x0b\x0a\x01\x00\x04more\x01\x7f\x00", "checksum"));
}

test "component resolver rejects wrong types options and forward references" {
    var changed = reference_component.*;
    changed[93] = 5; // list length
    try std.testing.expectError(error.UnsupportedComponentType, resolve(std.testing.allocator, &changed, "checksum"));
    changed = reference_component.*;
    changed[102] = 1; // function parameter type references itself
    try std.testing.expectError(error.InvalidComponentTypeIndex, resolve(std.testing.allocator, &changed, "checksum"));
    changed = reference_component.*;
    changed[127] = 1; // canonical option count
    try std.testing.expectError(error.UnsupportedCanonicalOptions, resolve(std.testing.allocator, &changed, "checksum"));
    changed = reference_component.*;
    changed[126] = 1; // nonexisting core function alias
    try std.testing.expectError(error.InvalidCanonicalFunctionIndex, resolve(std.testing.allocator, &changed, "checksum"));
    changed = reference_component.*;
    changed[111] = 1; // nonexisting instance
    try std.testing.expectError(error.InvalidComponentInstanceIndex, resolve(std.testing.allocator, &changed, "checksum"));
    // Instantiating before the module exists cannot resolve a forward reference.
    const forward = reference_component[0..8] ++ reference_component[82..88] ++ reference_component[8..82] ++ reference_component[88..];
    try std.testing.expectError(error.InvalidComponentModuleIndex, resolve(std.testing.allocator, forward, "checksum"));
}

test "component custom sections require valid names but preserve arbitrary payload" {
    _ = try resolve(std.testing.allocator, reference_component ++ "\x00\x06\x01x\xff\x07\x80\x00", "checksum");
    try std.testing.expectError(error.InvalidComponentUtf8, resolve(std.testing.allocator, reference_component ++ "\x00\x02\x01\xff", "checksum"));
    try std.testing.expectError(error.UnexpectedEndOfComponent, resolve(std.testing.allocator, reference_component ++ "\x00\x00", "checksum"));
    try std.testing.expectError(error.InvalidComponentLeb, resolve(std.testing.allocator, reference_component ++ "\x00\x80\x80\x80\x80\x10", "checksum"));
}

test "component resolver checks flattened signatures on every canonical lift" {
    const wrong_signature = reference_component[0..8] ++ "\x01\x2a" ++
        "\x00asm\x01\x00\x00\x00" ++
        "\x01\x06\x01\x60\x01\x7f\x01\x7f" ++
        "\x03\x02\x01\x00" ++
        "\x07\x0c\x01\x08checksum\x00\x00" ++
        "\x0a\x06\x01\x04\x00\x20\x00\x0b" ++
        reference_component[82..];
    try std.testing.expectError(error.CanonicalSignatureMismatch, resolve(std.testing.allocator, wrong_signature, "checksum"));
    try std.testing.expectError(error.UnsupportedComponentSignature, resolve(std.testing.allocator, reference_component ++ "\x08\x06\x01\x00\x00\x00\x00\x00", "checksum"));
    try std.testing.expectError(error.InvalidComponentTypeIndex, resolve(std.testing.allocator, reference_component ++ "\x08\x06\x01\x00\x00\x00\x00\x05", "checksum"));
}

test "component export ascriptions check parameter names and indices" {
    // A function type with a different parameter label has the same core shape,
    // but is not the same component function type.
    const prefix = reference_component ++ "\x07\x0c\x01\x40\x01\x05other\x00\x00\x79";
    try std.testing.expectError(error.ComponentExportTypeMismatch, resolve(std.testing.allocator, prefix ++ "\x0b\x0e\x01\x00\x06second\x01\x00\x01\x01\x02", "checksum"));
    const matching = try resolve(std.testing.allocator, reference_component ++ "\x0b\x0e\x01\x00\x06second\x01\x00\x01\x01\x01", "second");
    try std.testing.expectEqualStrings("checksum", matching.core_export);
}

test "core structural preflight rejects duplicate sections oversized locals and trailing bytes" {
    const header = "\x00asm\x01\x00\x00\x00";
    try std.testing.expectError(error.InvalidCoreSectionOrder, preflightCore(header ++ "\x01\x01\x00\x01\x01\x00"));
    try std.testing.expectError(error.InvalidCoreSectionOrder, preflightCore(header ++ "\x03\x01\x00\x01\x01\x00"));
    try std.testing.expectError(error.ComponentTooComplex, preflightCore(header ++ "\x0a\x08\x01\x06\x01\xff\xff\x7f\x7f\x0b"));
    try std.testing.expectError(error.TrailingCoreSectionBytes, preflightCore(header ++ "\x0a\x02\x00\x00"));
    try std.testing.expectError(error.ComponentImportsForbidden, preflightCore(header ++ "\x02\x01\x01"));
}

test "a list of unknown length resolves to the component's own memory and allocator" {
    const resolved = try resolve(std.testing.allocator, list_component, "checksum");
    try std.testing.expectEqualStrings("checksum", resolved.core_export);
    try std.testing.expectEqualStrings("cabi_realloc", resolved.lowering.memory.realloc_export);
    // The fixed-length shape stays flattened, with no memory and no allocator.
    const flattened = try resolve(std.testing.allocator, reference_component, "checksum");
    try std.testing.expectEqual(@as(u32, 4), flattened.lowering.flattened);
}

test "a lifted list needs both canonical options, and the core signature they imply" {
    // The canonical section holds: count, 0x00 0x00, core index, options, type.
    var changed = list_component.*;
    changed[358] = 1; // memory option index, with one aliased memory
    try std.testing.expectError(error.InvalidCanonicalMemoryIndex, resolve(std.testing.allocator, &changed, "checksum"));
    changed = list_component.*;
    changed[357] = 4; // a second realloc option in place of the memory
    try std.testing.expectError(error.DuplicateCanonicalOption, resolve(std.testing.allocator, &changed, "checksum"));
    changed = list_component.*;
    changed[355] = 0; // lift the allocator itself, whose core signature differs
    try std.testing.expectError(error.CanonicalSignatureMismatch, resolve(std.testing.allocator, &changed, "checksum"));

    // The same lift with the allocator alone: a list has nowhere to live.
    const without_memory = list_component[0..350] ++
        "\x08\x08\x01\x00\x00\x01\x01\x04\x00\x01" ++
        list_component[362..];
    try std.testing.expectError(error.MissingCanonicalMemory, resolve(std.testing.allocator, without_memory, "checksum"));
}

test "a list answer resolves to a return area the component releases" {
    const resolved = try resolve(std.testing.allocator, scan_component, "scan");
    try std.testing.expectEqualStrings("scan", resolved.core_export);
    try std.testing.expectEqualStrings("cabi_realloc", resolved.lowering.memory.realloc_export);
    try std.testing.expectEqualStrings("cabi_post_scan", resolved.lifting.memory.post_return_export.?);
    // A scalar answer rides in the core result, so it releases nothing.
    const scalar = try resolve(std.testing.allocator, list_component, "checksum");
    try std.testing.expectEqual(Lifting.scalar, scalar.lifting);
}
