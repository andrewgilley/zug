const std = @import("std");
const imports = @import("imports.zig");
const module = @import("module.zig");
const wasi_nn_abi = @import("../wasi_nn_abi.zig");

const wasm_page_size = 64 * 1024;
const max_wasm_pages = 65536;

pub const RuntimeGlobal = struct {
    global_type: module.GlobalType,
    value: module.ConstValue,
};

pub const RuntimeTable = struct {
    elements: []?u32,
};

pub const Instance = struct {
    allocator: std.mem.Allocator,
    module: *const module.Module,
    memory_bytes: []u8,
    memory: wasi_nn_abi.LinearMemory,
    import_resolver: ?*imports.Resolver = null,
    imported_functions: []imports.Function = &.{},
    globals: []RuntimeGlobal = &.{},
    tables: []RuntimeTable = &.{},
    start_executed: bool = false,

    pub fn init(
        allocator: std.mem.Allocator,
        parsed_module: *const module.Module,
        initial_memory_bytes: usize,
    ) !Instance {
        if (initial_memory_bytes == 0) return error.InvalidMemorySize;

        const memory_bytes = try allocator.alloc(
            u8,
            @max(initial_memory_bytes, try minimumMemoryBytes(parsed_module)),
        );
        errdefer allocator.free(memory_bytes);
        @memset(memory_bytes, 0);

        var result = Instance{
            .allocator = allocator,
            .module = parsed_module,
            .memory_bytes = memory_bytes,
            .memory = wasi_nn_abi.LinearMemory.init(memory_bytes),
        };
        errdefer result.deinit();

        try result.initGlobals();
        try result.initTables();
        try result.applyDataSegments();

        return result;
    }

    pub fn deinit(self: *Instance) void {
        if (self.imported_functions.len != 0) {
            self.allocator.free(self.imported_functions);
        }
        if (self.globals.len != 0) {
            self.allocator.free(self.globals);
        }
        for (self.tables) |table| {
            if (table.elements.len != 0) {
                self.allocator.free(table.elements);
            }
        }
        if (self.tables.len != 0) {
            self.allocator.free(self.tables);
        }
        self.allocator.free(self.memory_bytes);
        self.* = undefined;
    }

    pub fn bindImports(self: *Instance, resolver: *imports.Resolver) !void {
        if (self.imported_functions.len != 0) {
            self.allocator.free(self.imported_functions);
            self.imported_functions = &.{};
        }

        const count = self.importedFunctionCount();
        const resolved = try self.allocator.alloc(imports.Function, count);
        errdefer self.allocator.free(resolved);

        var index: usize = 0;
        for (self.module.imports.items) |import| {
            if (import.kind != .function) continue;

            resolved[index] = imports.Resolver.resolve(import.module, import.name) orelse {
                return error.UnresolvedFunctionImport;
            };
            index += 1;
        }

        self.imported_functions = resolved;
        self.import_resolver = resolver;
    }

    pub fn importedFunction(self: *const Instance, index: u32) !imports.Function {
        const actual = std.math.cast(usize, index) orelse return error.InvalidFunctionIndex;
        if (actual >= self.imported_functions.len) return error.InvalidFunctionIndex;

        return self.imported_functions[actual];
    }

    pub fn definedFunction(self: *const Instance, function_index: u32) !module.Function {
        const imported_count = self.importedFunctionCount();
        const actual = std.math.cast(usize, function_index) orelse return error.InvalidFunctionIndex;

        if (actual < imported_count) return error.ImportedFunctionIndex;

        const defined_index = actual - imported_count;
        if (defined_index >= self.module.functions.items.len) return error.InvalidFunctionIndex;

        return self.module.functions.items[defined_index];
    }

    pub fn functionType(self: *const Instance, type_index: u32) !module.FunctionType {
        return self.module.functionType(type_index);
    }

    pub fn importedFunctionType(self: *const Instance, function_index: u32) !module.FunctionType {
        return self.functionType(try self.functionTypeIndex(function_index));
    }

    pub fn functionTypeIndex(self: *const Instance, function_index: u32) !u32 {
        const actual = std.math.cast(usize, function_index) orelse return error.InvalidFunctionIndex;
        const imported_count = self.importedFunctionCount();

        if (actual < imported_count) {
            var current: usize = 0;

            for (self.module.imports.items) |import| {
                if (import.kind != .function) continue;

                if (current == actual) {
                    return import.type_index orelse return error.MissingImportTypeIndex;
                }

                current += 1;
            }

            return error.InvalidFunctionIndex;
        }

        const defined_index = actual - imported_count;
        if (defined_index >= self.module.functions.items.len) return error.InvalidFunctionIndex;

        return self.module.functions.items[defined_index].type_index;
    }

    pub fn exportedFunctionIndex(self: *const Instance, name: []const u8) !u32 {
        for (self.module.exports.items) |exported| {
            if (exported.kind == .function and std.mem.eql(u8, exported.name, name)) {
                return exported.index;
            }
        }

        return error.UnknownFunctionExport;
    }

    pub fn importedFunctionCount(self: *const Instance) usize {
        var count: usize = 0;

        for (self.module.imports.items) |import| {
            if (import.kind == .function) count += 1;
        }

        return count;
    }

    pub fn importedGlobalCount(self: *const Instance) usize {
        var count: usize = 0;

        for (self.module.imports.items) |import| {
            if (import.kind == .global) count += 1;
        }

        return count;
    }

    pub fn importedTableCount(self: *const Instance) usize {
        var count: usize = 0;

        for (self.module.imports.items) |import| {
            if (import.kind == .table) count += 1;
        }

        return count;
    }

    pub fn functionCount(self: *const Instance) usize {
        return self.importedFunctionCount() + self.module.functions.items.len;
    }

    pub fn tableFunctionIndex(self: *const Instance, table_index: u32, element_index: u32) !u32 {
        const actual_table = std.math.cast(usize, table_index) orelse return error.InvalidTableIndex;
        if (actual_table >= self.tables.len) return error.InvalidTableIndex;

        const actual_element = std.math.cast(usize, element_index) orelse return error.InvalidTableElementIndex;
        if (actual_element >= self.tables[actual_table].elements.len) return error.InvalidTableElementIndex;

        return self.tables[actual_table].elements[actual_element] orelse error.UninitializedTableElement;
    }

    pub fn global(self: *const Instance, index: u32) !RuntimeGlobal {
        const actual = std.math.cast(usize, index) orelse return error.InvalidGlobalIndex;
        const imported_count = self.importedGlobalCount();

        if (actual < imported_count) return error.ImportedGlobalsUnsupported;

        const defined_index = actual - imported_count;
        if (defined_index >= self.globals.len) return error.InvalidGlobalIndex;

        return self.globals[defined_index];
    }

    pub fn setGlobalValue(self: *Instance, index: u32, value: module.ConstValue) !void {
        const actual = std.math.cast(usize, index) orelse return error.InvalidGlobalIndex;
        const imported_count = self.importedGlobalCount();

        if (actual < imported_count) return error.ImportedGlobalsUnsupported;

        const defined_index = actual - imported_count;
        if (defined_index >= self.globals.len) return error.InvalidGlobalIndex;
        if (!self.globals[defined_index].global_type.mutable) return error.ImmutableGlobal;
        if (!constValueMatchesType(value, self.globals[defined_index].global_type.value_type)) {
            return error.GlobalTypeMismatch;
        }

        self.globals[defined_index].value = value;
    }

    pub fn startFunctionIndex(self: *const Instance) ?u32 {
        return self.module.start_function_index;
    }

    pub fn currentMemoryPages(self: *const Instance) !u32 {
        if (self.module.memories.items.len != 1) return error.MemoryUnsupported;
        if (self.memory_bytes.len % wasm_page_size != 0) return error.InvalidMemorySize;

        const pages = self.memory_bytes.len / wasm_page_size;
        return std.math.cast(u32, pages) orelse error.MemorySizeTooLarge;
    }

    pub fn growMemory(self: *Instance, delta_pages: u32) !?u32 {
        const old_pages = try self.currentMemoryPages();
        const new_pages = std.math.add(u32, old_pages, delta_pages) catch return null;
        const max_pages = try self.memoryMaxPages();

        if (new_pages > max_pages) return null;
        if (delta_pages == 0) return old_pages;

        const new_len = std.math.mul(usize, new_pages, wasm_page_size) catch return null;
        const old_len = self.memory_bytes.len;
        const grown = self.allocator.realloc(self.memory_bytes, new_len) catch return null;

        @memset(grown[old_len..], 0);

        self.memory_bytes = grown;
        self.memory = wasi_nn_abi.LinearMemory.init(self.memory_bytes);

        return old_pages;
    }

    fn initGlobals(self: *Instance) !void {
        if (self.importedGlobalCount() != 0) return error.ImportedGlobalsUnsupported;
        if (self.module.globals.items.len == 0) return;

        const globals = try self.allocator.alloc(RuntimeGlobal, self.module.globals.items.len);
        errdefer self.allocator.free(globals);

        for (self.module.globals.items, 0..) |global_def, index| {
            globals[index] = .{
                .global_type = global_def.global_type,
                .value = global_def.init,
            };
        }

        self.globals = globals;
    }

    fn initTables(self: *Instance) !void {
        if (self.importedTableCount() != 0) return error.ImportedTablesUnsupported;
        if (self.module.tables.items.len == 0) {
            if (self.module.element_segments.items.len != 0) return error.MissingTable;
            return;
        }

        const tables = try self.allocator.alloc(RuntimeTable, self.module.tables.items.len);
        for (tables) |*table| {
            table.* = .{ .elements = &.{} };
        }
        self.tables = tables;

        for (self.module.tables.items, 0..) |table_def, index| {
            const len = std.math.cast(usize, table_def.limits.min) orelse return error.TableTooLarge;
            const elements = try self.allocator.alloc(?u32, len);
            for (elements) |*element| {
                element.* = null;
            }
            self.tables[index].elements = elements;
        }

        try self.applyElementSegments();
    }

    fn applyElementSegments(self: *Instance) !void {
        for (self.module.element_segments.items) |segment| {
            const actual_table = std.math.cast(usize, segment.table_index) orelse return error.InvalidTableIndex;
            if (actual_table >= self.tables.len) return error.InvalidTableIndex;

            const start = std.math.cast(usize, segment.offset) orelse return error.InvalidTableElementIndex;
            const end = try std.math.add(usize, start, segment.function_indices.len);
            if (end > self.tables[actual_table].elements.len) return error.InvalidTableElementIndex;

            for (segment.function_indices, 0..) |function_index, index| {
                const actual_function = std.math.cast(usize, function_index) orelse return error.InvalidFunctionIndex;
                if (actual_function >= self.functionCount()) return error.InvalidFunctionIndex;

                self.tables[actual_table].elements[start + index] = function_index;
            }
        }
    }

    fn memoryMaxPages(self: *const Instance) !u32 {
        if (self.module.memories.items.len != 1) return error.MemoryUnsupported;
        return self.module.memories.items[0].limits.max orelse max_wasm_pages;
    }

    fn applyDataSegments(self: *Instance) !void {
        for (self.module.data_segments.items) |segment| {
            if (segment.memory_index != 0) return error.UnsupportedMemoryIndex;
            try self.memory.write(segment.offset, segment.bytes);
        }
    }
};

fn constValueMatchesType(value: module.ConstValue, value_type: module.ValueType) bool {
    return switch (value_type) {
        .i32 => std.meta.activeTag(value) == .i32,
        .i64 => std.meta.activeTag(value) == .i64,
        .f32 => std.meta.activeTag(value) == .f32,
        .f64 => std.meta.activeTag(value) == .f64,
    };
}

fn minimumMemoryBytes(parsed_module: *const module.Module) !usize {
    if (parsed_module.memories.items.len == 0) return 0;
    if (parsed_module.memories.items.len > 1) return error.MultipleMemoriesUnsupported;

    const pages = std.math.cast(usize, parsed_module.memories.items[0].limits.min) orelse {
        return error.MemorySizeTooLarge;
    };

    return try std.math.mul(usize, pages, wasm_page_size);
}

test "instance owns linear memory" {
    const allocator = std.testing.allocator;

    var parsed = try module.Module.parse(allocator, "\x00asm\x01\x00\x00\x00");
    defer parsed.deinit(allocator);

    var instance = try Instance.init(allocator, &parsed, 64 * 1024);
    defer instance.deinit();

    try instance.memory.writeU32(0, 42);
    try std.testing.expectEqual(@as(u32, 42), try instance.memory.readU32(0));
}

test "instance resolves wasi-nn function imports" {
    const allocator = std.testing.allocator;

    const bytes =
        "\x00asm\x01\x00\x00\x00" ++
        "\x02\x13" ++
        "\x01" ++
        "\x07wasi_nn" ++
        "\x07compute" ++
        "\x00" ++
        "\x00";

    var parsed = try module.Module.parse(allocator, bytes);
    defer parsed.deinit(allocator);

    var wasm_instance = try Instance.init(allocator, &parsed, 64 * 1024);
    defer wasm_instance.deinit();

    var host = wasi_nn_abi.Host.init(allocator);
    defer host.deinit();

    var surface = wasi_nn_abi.Surface.init(allocator, &host, &wasm_instance.memory);
    var resolver = imports.Resolver.init(&surface);

    try wasm_instance.bindImports(&resolver);

    try std.testing.expectEqual(@as(usize, 1), wasm_instance.importedFunctionCount());
    try std.testing.expectEqual(imports.Function.compute, try wasm_instance.importedFunction(0));
}

test "instance applies active data segments" {
    const allocator = std.testing.allocator;

    const bytes =
        "\x00asm\x01\x00\x00\x00" ++
        "\x05\x03" ++
        "\x01\x00\x01" ++
        "\x0b\x09" ++
        "\x01\x00\x41\x04\x0b\x03abc";

    var parsed = try module.Module.parse(allocator, bytes);
    defer parsed.deinit(allocator);

    var wasm_instance = try Instance.init(allocator, &parsed, 16);
    defer wasm_instance.deinit();

    try std.testing.expectEqualSlices(u8, "abc", try wasm_instance.memory.read(4, 3));
    try std.testing.expectEqual(@as(usize, wasm_page_size), wasm_instance.memory_bytes.len);
}

test "instance grows memory within declared maximum" {
    const allocator = std.testing.allocator;

    const bytes =
        "\x00asm\x01\x00\x00\x00" ++
        "\x05\x04" ++
        "\x01\x01\x01\x03";

    var parsed = try module.Module.parse(allocator, bytes);
    defer parsed.deinit(allocator);

    var wasm_instance = try Instance.init(allocator, &parsed, 64 * 1024);
    defer wasm_instance.deinit();

    try std.testing.expectEqual(@as(u32, 1), try wasm_instance.currentMemoryPages());
    try std.testing.expectEqual(@as(?u32, 1), try wasm_instance.growMemory(1));
    try std.testing.expectEqual(@as(u32, 2), try wasm_instance.currentMemoryPages());
    try std.testing.expectEqual(@as(?u32, null), try wasm_instance.growMemory(2));
    try std.testing.expectEqual(@as(u32, 2), try wasm_instance.currentMemoryPages());
}

test "instance initializes and mutates defined globals" {
    const allocator = std.testing.allocator;

    const bytes =
        "\x00asm\x01\x00\x00\x00" ++
        "\x06\x06" ++
        "\x01\x7f\x01\x41\x07\x0b";

    var parsed = try module.Module.parse(allocator, bytes);
    defer parsed.deinit(allocator);

    var wasm_instance = try Instance.init(allocator, &parsed, 64 * 1024);
    defer wasm_instance.deinit();

    try std.testing.expectEqual(@as(usize, 1), wasm_instance.globals.len);
    try std.testing.expectEqual(@as(u32, 7), switch ((try wasm_instance.global(0)).value) {
        .i32 => |value| value,
        else => return error.ExpectedI32Global,
    });

    try wasm_instance.setGlobalValue(0, .{ .i32 = 12 });
    try std.testing.expectEqual(@as(u32, 12), switch ((try wasm_instance.global(0)).value) {
        .i32 => |value| value,
        else => return error.ExpectedI32Global,
    });
}

test "instance initializes active table element segments" {
    const allocator = std.testing.allocator;

    var parsed = try module.Module.parse(allocator, @import("fixtures.zig").indirect_function_call);
    defer parsed.deinit(allocator);

    var wasm_instance = try Instance.init(allocator, &parsed, 64 * 1024);
    defer wasm_instance.deinit();

    try std.testing.expectEqual(@as(usize, 1), wasm_instance.tables.len);
    try std.testing.expectEqual(@as(usize, 2), wasm_instance.tables[0].elements.len);
    try std.testing.expectEqual(@as(u32, 0), try wasm_instance.tableFunctionIndex(0, 0));
    try std.testing.expectEqual(@as(u32, 1), try wasm_instance.tableFunctionIndex(0, 1));
}
