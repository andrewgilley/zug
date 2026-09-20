const std = @import("std");
const imports = @import("imports.zig");
const memory_module = @import("memory.zig");
const module = @import("module.zig");

const wasm_page_size = 64 * 1024;
const max_wasm_pages = 65536;

pub const RuntimeGlobal = struct {
    global_type: module.GlobalType,
    value: module.ConstValue,
};

pub const RuntimeTable = struct {
    elements: []?u32,
};

const IndexRange = struct {
    start: usize,
    end: usize,
    len: usize,
};

/// Linear memory, owned by the instance whose module defines it and shared
/// with the instances that import it. The view lives behind this pointer, so
/// growth by one holder is visible to every other.
pub const Memory = struct {
    allocator: std.mem.Allocator,
    view: memory_module.LinearMemory,
    /// The defining module's declared maximum, in pages.
    max_pages: u32,

    pub fn create(
        allocator: std.mem.Allocator,
        initial_bytes: usize,
        max_pages: u32,
    ) !*Memory {
        const bytes = try allocator.alloc(u8, initial_bytes);
        @memset(bytes, 0);

        const self = allocator.create(Memory) catch |err| {
            allocator.free(bytes);
            return err;
        };
        self.* = .{
            .allocator = allocator,
            .view = memory_module.LinearMemory.init(bytes),
            .max_pages = max_pages,
        };
        return self;
    }

    pub fn destroy(self: *Memory) void {
        const allocator = self.allocator;
        allocator.free(self.view.bytes);
        allocator.destroy(self);
    }
};

/// Where a function import's calls go: a WASI host function, or a function
/// exported by another instance (core-module linking).
pub const ImportBinding = union(enum) {
    host: imports.Function,
    linked: LinkedFunction,
};

pub const LinkedFunction = struct {
    instance: *Instance,
    function_index: u32,
};

/// An earlier instance whose function exports later instances may import by
/// using its name as the import module.
pub const NamedInstance = struct {
    name: []const u8,
    instance: *Instance,
};

pub const Instance = struct {
    allocator: std.mem.Allocator,
    module: *const module.Module,
    /// Owned when this module defines its memory, borrowed when it imports one.
    memory_ref: *Memory,
    owns_memory: bool,
    import_resolver: ?*imports.Resolver = null,
    imported_functions: []ImportBinding = &.{},
    globals: []RuntimeGlobal = &.{},
    tables: []RuntimeTable = &.{},
    dropped_element_segments: []bool = &.{},
    dropped_data_segments: []bool = &.{},
    start_executed: bool = false,
    /// Optional host budget; memory.grow returns -1 when it would exceed this.
    max_memory_bytes: ?usize = null,

    pub fn init(
        allocator: std.mem.Allocator,
        parsed_module: *const module.Module,
        initial_memory_bytes: usize,
    ) !Instance {
        return initWithMemory(allocator, parsed_module, initial_memory_bytes, null);
    }

    /// Instantiate against a memory another instance exports. `imported` must be
    /// present exactly when the module imports its memory; `resolveImportedMemory`
    /// finds it and reports the link failures before anything is instantiated.
    pub fn initWithMemory(
        allocator: std.mem.Allocator,
        parsed_module: *const module.Module,
        initial_memory_bytes: usize,
        imported: ?*Memory,
    ) !Instance {
        if ((importedMemory(parsed_module) != null) != (imported != null)) {
            return error.MemoryImportMismatch;
        }

        var result = Instance{
            .allocator = allocator,
            .module = parsed_module,
            .memory_ref = imported orelse try Memory.create(
                allocator,
                @max(initial_memory_bytes, try minimumMemoryBytes(parsed_module)),
                declaredMaxPages(parsed_module),
            ),
            .owns_memory = imported == null,
        };
        errdefer result.deinit();

        try result.initGlobals();
        try result.initTables();
        try result.applyDataSegments();

        return result;
    }

    /// The linear memory this instance reads and writes, shared with the
    /// instance that exports it when the module imports its memory.
    pub fn memory(self: *const Instance) *memory_module.LinearMemory {
        return &self.memory_ref.view;
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
        if (self.dropped_element_segments.len != 0) {
            self.allocator.free(self.dropped_element_segments);
        }
        if (self.dropped_data_segments.len != 0) {
            self.allocator.free(self.dropped_data_segments);
        }
        if (self.owns_memory) {
            self.memory_ref.destroy();
        }
        self.* = undefined;
    }

    pub fn bindImports(self: *Instance, resolver: *imports.Resolver) !void {
        if (self.imported_functions.len != 0) {
            self.allocator.free(self.imported_functions);
            self.imported_functions = &.{};
        }

        const count = self.importedFunctionCount();
        const resolved = try self.allocator.alloc(ImportBinding, count);
        errdefer self.allocator.free(resolved);

        var index: usize = 0;
        for (self.module.imports.items) |import| {
            if (import.kind != .function) continue;

            resolved[index] = .{ .host = imports.Resolver.resolve(import.module, import.name) orelse {
                return error.UnresolvedFunctionImport;
            } };
            index += 1;
        }

        self.imported_functions = resolved;
        self.import_resolver = resolver;
    }

    /// Bind every function import to an export of an earlier instance whose name
    /// matches the import module. Nothing is granted from the host, and only
    /// function and memory imports are supported; the memory is bound earlier,
    /// by `resolveImportedMemory`, because data segments need it.
    pub fn bindLinkedImports(self: *Instance, providers: []const NamedInstance) !void {
        if (self.imported_functions.len != 0) {
            self.allocator.free(self.imported_functions);
            self.imported_functions = &.{};
        }

        const resolved = try self.allocator.alloc(ImportBinding, self.importedFunctionCount());
        errdefer self.allocator.free(resolved);

        var index: usize = 0;
        for (self.module.imports.items) |import| {
            // The memory import, if any, was bound before instantiation.
            if (import.kind == .memory) continue;
            if (import.kind != .function) return error.UnsupportedLinkedImportKind;

            const provider = for (providers) |candidate| {
                if (std.mem.eql(u8, candidate.name, import.module)) break candidate.instance;
            } else return error.UnresolvedFunctionImport;

            const function_index = provider.exportedFunctionIndex(import.name) catch {
                return error.UnresolvedFunctionImport;
            };
            const expected = try self.functionType(import.type_index orelse return error.MissingImportTypeIndex);
            const actual = try provider.functionType(try provider.functionTypeIndex(function_index));
            if (!std.mem.eql(module.ValueType, expected.params, actual.params) or
                !std.mem.eql(module.ValueType, expected.results, actual.results))
            {
                return error.IncompatibleImportType;
            }

            resolved[index] = .{ .linked = .{ .instance = provider, .function_index = function_index } };
            index += 1;
        }

        self.imported_functions = resolved;
    }

    pub fn importedFunction(self: *const Instance, index: u32) !ImportBinding {
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

    pub fn tableInit(
        self: *Instance,
        element_index: u32,
        table_index: u32,
        destination: u32,
        source: u32,
        len: u32,
    ) !void {
        const segment = try self.elementSegment(element_index);
        if (try self.elementSegmentDropped(element_index)) return error.ElementSegmentDropped;

        const actual_table = std.math.cast(usize, table_index) orelse return error.InvalidTableIndex;
        if (actual_table >= self.tables.len) return error.InvalidTableIndex;

        const source_range = try checkedRange(segment.function_indices.len, source, len);
        const destination_range = try checkedRange(self.tables[actual_table].elements.len, destination, len);

        for (
            self.tables[actual_table].elements[destination_range.start..destination_range.end],
            segment.function_indices[source_range.start..source_range.end],
        ) |*slot, function_index| {
            const actual_function = std.math.cast(usize, function_index) orelse return error.InvalidFunctionIndex;
            if (actual_function >= self.functionCount()) return error.InvalidFunctionIndex;
            slot.* = function_index;
        }
    }

    pub fn tableCopy(
        self: *Instance,
        destination_table_index: u32,
        source_table_index: u32,
        destination: u32,
        source: u32,
        len: u32,
    ) !void {
        const destination_table = try self.tableIndex(destination_table_index);
        const source_table = try self.tableIndex(source_table_index);
        const source_range = try checkedRange(self.tables[source_table].elements.len, source, len);
        const destination_range = try checkedRange(self.tables[destination_table].elements.len, destination, len);

        if (destination_table == source_table and destination_range.start > source_range.start) {
            var index = source_range.len;
            while (index > 0) {
                index -= 1;
                self.tables[destination_table].elements[destination_range.start + index] =
                    self.tables[source_table].elements[source_range.start + index];
            }
        } else {
            for (0..source_range.len) |index| {
                self.tables[destination_table].elements[destination_range.start + index] =
                    self.tables[source_table].elements[source_range.start + index];
            }
        }
    }

    pub fn tableSize(self: *const Instance, table_index: u32) !u32 {
        const actual_table = try self.tableIndex(table_index);
        return std.math.cast(u32, self.tables[actual_table].elements.len) orelse error.TableTooLarge;
    }

    pub fn tableGrow(self: *Instance, table_index: u32, value: ?u32, delta: u32) !?u32 {
        try self.validateFunctionRef(value);

        const actual_table = try self.tableIndex(table_index);
        const old_size = try self.tableSize(table_index);
        const new_size = std.math.add(u32, old_size, delta) catch return null;
        const max_size = self.module.tables.items[actual_table].limits.max orelse std.math.maxInt(u32);
        if (new_size > max_size) return null;
        if (delta == 0) return old_size;

        const new_len = std.math.cast(usize, new_size) orelse return null;
        const old_len = self.tables[actual_table].elements.len;
        const grown = self.allocator.realloc(self.tables[actual_table].elements, new_len) catch return null;
        for (grown[old_len..]) |*slot| {
            slot.* = value;
        }
        self.tables[actual_table].elements = grown;

        return old_size;
    }

    pub fn tableFill(
        self: *Instance,
        table_index: u32,
        destination: u32,
        value: ?u32,
        len: u32,
    ) !void {
        try self.validateFunctionRef(value);

        const actual_table = try self.tableIndex(table_index);
        const range = try checkedRange(self.tables[actual_table].elements.len, destination, len);
        for (self.tables[actual_table].elements[range.start..range.end]) |*slot| {
            slot.* = value;
        }
    }

    pub fn elementDrop(self: *Instance, element_index: u32) !void {
        _ = try self.elementSegment(element_index);

        const actual = std.math.cast(usize, element_index) orelse return error.InvalidElementSegmentIndex;
        self.dropped_element_segments[actual] = true;
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
        if (memoryCount(self.module) != 1) return error.MemoryUnsupported;
        if (self.memory().bytes.len % wasm_page_size != 0) return error.InvalidMemorySize;

        const pages = self.memory().bytes.len / wasm_page_size;
        return std.math.cast(u32, pages) orelse error.MemorySizeTooLarge;
    }

    pub fn growMemory(self: *Instance, delta_pages: u32) !?u32 {
        const old_pages = try self.currentMemoryPages();
        const new_pages = std.math.add(u32, old_pages, delta_pages) catch return null;
        const max_pages = try self.memoryMaxPages();

        if (new_pages > max_pages) return null;
        if (delta_pages == 0) return old_pages;

        const new_len = std.math.mul(usize, new_pages, wasm_page_size) catch return null;
        if (self.max_memory_bytes) |limit| {
            if (new_len > limit) return null;
        }
        const old_len = self.memory().bytes.len;
        // The owning allocator grows the shared buffer, so every holder of the
        // memory sees the new pages through the same view.
        const grown = self.memory_ref.allocator.realloc(self.memory().bytes, new_len) catch return null;

        @memset(grown[old_len..], 0);

        self.memory_ref.view = memory_module.LinearMemory.init(grown);

        return old_pages;
    }

    pub fn memoryInit(
        self: *Instance,
        data_index: u32,
        destination: u32,
        source: u32,
        len: u32,
    ) !void {
        const segment = try self.dataSegment(data_index);
        if (try self.dataSegmentDropped(data_index)) return error.DataSegmentDropped;

        const source_range = try checkedRange(segment.bytes.len, source, len);
        const destination_range = try checkedRange(self.memory().bytes.len, destination, len);

        @memcpy(
            self.memory().bytes[destination_range.start..destination_range.end],
            segment.bytes[source_range.start..source_range.end],
        );
    }

    pub fn dataDrop(self: *Instance, data_index: u32) !void {
        _ = try self.dataSegment(data_index);

        const actual = std.math.cast(usize, data_index) orelse return error.InvalidDataSegmentIndex;
        self.dropped_data_segments[actual] = true;
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
        try self.initElementSegmentState();

        if (self.module.tables.items.len == 0) {
            for (self.module.element_segments.items) |segment| {
                if (!segment.passive) return error.MissingTable;
            }
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

    fn initDataSegmentState(self: *Instance) !void {
        if (self.module.data_segments.items.len == 0) return;

        const dropped = try self.allocator.alloc(bool, self.module.data_segments.items.len);
        @memset(dropped, false);
        self.dropped_data_segments = dropped;
    }

    fn initElementSegmentState(self: *Instance) !void {
        if (self.module.element_segments.items.len == 0) return;

        const dropped = try self.allocator.alloc(bool, self.module.element_segments.items.len);
        @memset(dropped, false);
        self.dropped_element_segments = dropped;
    }

    fn applyElementSegments(self: *Instance) !void {
        for (self.module.element_segments.items) |segment| {
            if (segment.passive) continue;

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
        if (memoryCount(self.module) != 1) return error.MemoryUnsupported;
        return self.memory_ref.max_pages;
    }

    fn applyDataSegments(self: *Instance) !void {
        try self.initDataSegmentState();

        for (self.module.data_segments.items) |segment| {
            if (segment.passive) continue;
            if (segment.memory_index != 0) return error.UnsupportedMemoryIndex;
            try self.memory().write(segment.offset, segment.bytes);
        }
    }

    fn dataSegment(self: *const Instance, data_index: u32) !module.DataSegment {
        const actual = std.math.cast(usize, data_index) orelse return error.InvalidDataSegmentIndex;
        if (actual >= self.module.data_segments.items.len) return error.InvalidDataSegmentIndex;

        return self.module.data_segments.items[actual];
    }

    fn dataSegmentDropped(self: *const Instance, data_index: u32) !bool {
        const actual = std.math.cast(usize, data_index) orelse return error.InvalidDataSegmentIndex;
        if (actual >= self.dropped_data_segments.len) return error.InvalidDataSegmentIndex;

        return self.dropped_data_segments[actual];
    }

    fn elementSegment(self: *const Instance, element_index: u32) !module.ElementSegment {
        const actual = std.math.cast(usize, element_index) orelse return error.InvalidElementSegmentIndex;
        if (actual >= self.module.element_segments.items.len) return error.InvalidElementSegmentIndex;

        return self.module.element_segments.items[actual];
    }

    fn elementSegmentDropped(self: *const Instance, element_index: u32) !bool {
        const actual = std.math.cast(usize, element_index) orelse return error.InvalidElementSegmentIndex;
        if (actual >= self.dropped_element_segments.len) return error.InvalidElementSegmentIndex;

        return self.dropped_element_segments[actual];
    }

    fn tableIndex(self: *const Instance, table_index: u32) !usize {
        const actual_table = std.math.cast(usize, table_index) orelse return error.InvalidTableIndex;
        if (actual_table >= self.tables.len) return error.InvalidTableIndex;

        return actual_table;
    }

    fn validateFunctionRef(self: *const Instance, value: ?u32) !void {
        if (value) |function_index| {
            const actual_function = std.math.cast(usize, function_index) orelse return error.InvalidFunctionIndex;
            if (actual_function >= self.functionCount()) return error.InvalidFunctionIndex;
        }
    }
};

fn checkedRange(container_len: usize, start_raw: u32, len_raw: u32) !IndexRange {
    const start = std.math.cast(usize, start_raw) orelse return error.InvalidMemoryRange;
    const len = std.math.cast(usize, len_raw) orelse return error.InvalidMemoryRange;
    const end = try std.math.add(usize, start, len);
    if (end > container_len) return error.InvalidMemoryRange;

    return .{ .start = start, .end = end, .len = len };
}

fn constValueMatchesType(value: module.ConstValue, value_type: module.ValueType) bool {
    return switch (value_type) {
        .funcref => std.meta.activeTag(value) == .funcref,
        .i32 => std.meta.activeTag(value) == .i32,
        .i64 => std.meta.activeTag(value) == .i64,
        .f32 => std.meta.activeTag(value) == .f32,
        .f64 => std.meta.activeTag(value) == .f64,
        .v128 => std.meta.activeTag(value) == .v128,
    };
}

fn minimumMemoryBytes(parsed_module: *const module.Module) !usize {
    if (parsed_module.memories.items.len == 0) return 0;
    if (memoryCount(parsed_module) > 1) return error.MultipleMemoriesUnsupported;

    const pages = std.math.cast(usize, parsed_module.memories.items[0].limits.min) orelse {
        return error.MemorySizeTooLarge;
    };

    return try std.math.mul(usize, pages, wasm_page_size);
}

/// Defined and imported memories share one index space, so both count.
pub fn memoryCount(parsed_module: *const module.Module) usize {
    return parsed_module.memories.items.len +
        @as(usize, if (importedMemory(parsed_module) == null) 0 else 1);
}

pub fn importedMemory(parsed_module: *const module.Module) ?module.Import {
    for (parsed_module.imports.items) |import| {
        if (import.kind == .memory) return import;
    }
    return null;
}

fn declaredMaxPages(parsed_module: *const module.Module) u32 {
    if (parsed_module.memories.items.len != 1) return max_wasm_pages;
    return parsed_module.memories.items[0].limits.max orelse max_wasm_pages;
}

/// Bind a module's memory import to the memory an earlier instance exports.
/// Limits match the core Wasm rule: the provider must offer at least the pages
/// the importer requires, and may not grow past a maximum the importer declares.
pub fn resolveImportedMemory(
    parsed_module: *const module.Module,
    providers: []const NamedInstance,
) !?*Memory {
    const import = importedMemory(parsed_module) orelse return null;
    if (memoryCount(parsed_module) > 1) return error.MultipleMemoriesUnsupported;
    const limits = import.limits orelse return error.MissingImportLimits;

    const provider = for (providers) |candidate| {
        if (std.mem.eql(u8, candidate.name, import.module)) break candidate.instance;
    } else return error.UnresolvedMemoryImport;

    var exported = false;
    for (provider.module.exports.items) |candidate| {
        if (candidate.kind == .memory and candidate.index == 0 and
            std.mem.eql(u8, candidate.name, import.name)) exported = true;
    }
    if (!exported) return error.UnresolvedMemoryImport;

    const available = try provider.currentMemoryPages();
    if (available < limits.min) return error.IncompatibleMemoryLimits;
    if (limits.max) |ceiling| {
        if (provider.memory_ref.max_pages > ceiling) return error.IncompatibleMemoryLimits;
    }

    return provider.memory_ref;
}

test "instance owns linear memory" {
    const allocator = std.testing.allocator;

    var parsed = try module.Module.parse(allocator, "\x00asm\x01\x00\x00\x00");
    defer parsed.deinit(allocator);

    var instance = try Instance.init(allocator, &parsed, 64 * 1024);
    defer instance.deinit();

    try instance.memory().writeU32(0, 42);
    try std.testing.expectEqual(@as(u32, 42), try instance.memory().readU32(0));
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

    try std.testing.expectEqualSlices(u8, "abc", try wasm_instance.memory().read(4, 3));
    try std.testing.expectEqual(@as(usize, wasm_page_size), wasm_instance.memory().bytes.len);
}

test "instance leaves passive data for explicit memory init" {
    const allocator = std.testing.allocator;

    var parsed = try module.Module.parse(allocator, @import("fixtures.zig").passive_data_memory_init);
    defer parsed.deinit(allocator);

    var wasm_instance = try Instance.init(allocator, &parsed, 64 * 1024);
    defer wasm_instance.deinit();

    try std.testing.expectEqualSlices(u8, &.{ 0, 0, 0, 0 }, try wasm_instance.memory().read(8, 4));

    try wasm_instance.memoryInit(0, 8, 1, 4);
    try std.testing.expectEqualSlices(u8, "WASM", try wasm_instance.memory().read(8, 4));

    try wasm_instance.dataDrop(0);
    try std.testing.expectError(error.DataSegmentDropped, wasm_instance.memoryInit(0, 16, 0, 1));
}

test "instance initializes tables from passive element segments" {
    const allocator = std.testing.allocator;

    var parsed = try module.Module.parse(allocator, @import("fixtures.zig").passive_element_table_init);
    defer parsed.deinit(allocator);

    var wasm_instance = try Instance.init(allocator, &parsed, 64 * 1024);
    defer wasm_instance.deinit();

    try std.testing.expectError(error.UninitializedTableElement, wasm_instance.tableFunctionIndex(0, 0));

    try wasm_instance.tableInit(0, 0, 0, 0, 2);
    try std.testing.expectEqual(@as(u32, 1), try wasm_instance.tableFunctionIndex(0, 0));
    try std.testing.expectEqual(@as(u32, 2), try wasm_instance.tableFunctionIndex(0, 1));

    try wasm_instance.elementDrop(0);
    try std.testing.expectError(error.ElementSegmentDropped, wasm_instance.tableInit(0, 0, 0, 0, 1));
}

test "instance copies grows sizes and fills tables" {
    const allocator = std.testing.allocator;

    var parsed = try module.Module.parse(allocator, @import("fixtures.zig").table_copy_grow_size_fill);
    defer parsed.deinit(allocator);

    var wasm_instance = try Instance.init(allocator, &parsed, 64 * 1024);
    defer wasm_instance.deinit();

    try wasm_instance.tableInit(0, 0, 0, 0, 2);
    try std.testing.expectEqual(@as(u32, 1), try wasm_instance.tableFunctionIndex(0, 0));
    try std.testing.expectEqual(@as(u32, 2), try wasm_instance.tableFunctionIndex(0, 1));

    try wasm_instance.tableCopy(0, 0, 0, 1, 1);
    try std.testing.expectEqual(@as(u32, 2), try wasm_instance.tableFunctionIndex(0, 0));
    try std.testing.expectEqual(@as(u32, 2), try wasm_instance.tableSize(0));

    try std.testing.expectEqual(@as(?u32, 2), try wasm_instance.tableGrow(0, 1, 1));
    try std.testing.expectEqual(@as(?u32, 3), try wasm_instance.tableGrow(0, null, 1));
    try std.testing.expectEqual(@as(?u32, null), try wasm_instance.tableGrow(0, null, 1));
    try std.testing.expectEqual(@as(u32, 4), try wasm_instance.tableSize(0));

    try wasm_instance.tableFill(0, 2, 1, 2);
    try std.testing.expectEqual(@as(u32, 1), try wasm_instance.tableFunctionIndex(0, 2));
    try std.testing.expectEqual(@as(u32, 1), try wasm_instance.tableFunctionIndex(0, 3));
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

test "an importing instance shares the exporter's memory, including growth" {
    const allocator = std.testing.allocator;
    const fixtures = @import("fixtures.zig");

    var provider_module = try module.Module.parse(allocator, fixtures.shared_memory_provider);
    defer provider_module.deinit(allocator);
    var consumer_module = try module.Module.parse(allocator, fixtures.shared_memory_consumer);
    defer consumer_module.deinit(allocator);

    var provider = try Instance.init(allocator, &provider_module, 64 * 1024);
    defer provider.deinit();

    const shared = (try resolveImportedMemory(&consumer_module, &.{.{ .name = "provider", .instance = &provider }})).?;
    var consumer = try Instance.initWithMemory(allocator, &consumer_module, 0, shared);
    defer consumer.deinit();

    // The consumer's data segment was written into the provider's memory.
    try std.testing.expectEqualSlices(u8, &.{ 1, 2, 3, 4 }, try provider.memory().read(64, 4));

    try consumer.memory().write(8, &.{ 9, 9 });
    try std.testing.expectEqualSlices(u8, &.{ 9, 9 }, try provider.memory().read(8, 2));

    // Growth reallocates the shared buffer, so both views follow it.
    try std.testing.expectEqual(@as(?u32, 1), try consumer.growMemory(1));
    try std.testing.expectEqual(@as(u32, 2), try provider.currentMemoryPages());
    try std.testing.expectEqual(@as(u32, 2), try consumer.currentMemoryPages());
    try std.testing.expectEqualSlices(u8, &.{ 9, 9 }, try provider.memory().read(8, 2));
}

test "an unmatched memory import is a link failure, not an instantiation" {
    const allocator = std.testing.allocator;
    const fixtures = @import("fixtures.zig");

    var provider_module = try module.Module.parse(allocator, fixtures.shared_memory_provider);
    defer provider_module.deinit(allocator);
    var consumer_module = try module.Module.parse(allocator, fixtures.shared_memory_consumer_two_pages);
    defer consumer_module.deinit(allocator);

    var provider = try Instance.init(allocator, &provider_module, 64 * 1024);
    defer provider.deinit();
    const providers = [_]NamedInstance{.{ .name = "provider", .instance = &provider }};

    try std.testing.expectError(
        error.IncompatibleMemoryLimits,
        resolveImportedMemory(&consumer_module, &providers),
    );
    try std.testing.expectError(
        error.UnresolvedMemoryImport,
        resolveImportedMemory(&consumer_module, &.{}),
    );
    // Instantiating a module that imports memory without one is refused.
    try std.testing.expectError(
        error.MemoryImportMismatch,
        Instance.initWithMemory(allocator, &consumer_module, 0, null),
    );
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
