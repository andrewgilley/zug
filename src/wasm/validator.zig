const std = @import("std");
const binary = @import("binary.zig");
const fixtures = @import("fixtures.zig");
const module = @import("module.zig");

const wasm_page_size = 64 * 1024;

const BlockKind = enum {
    block,
    loop,
};

const BlockType = struct {
    result: ?module.ValueType = null,

    fn eql(lhs: BlockType, rhs: BlockType) bool {
        return lhs.result == rhs.result;
    }
};

const Flow = union(enum) {
    branch: u32,
    returned,
    trapped,
};

pub fn validate(allocator: std.mem.Allocator, parsed: *const module.Module) !void {
    var validator = Validator{
        .allocator = allocator,
        .parsed = parsed,
    };

    try validator.validateModule();
}

const Validator = struct {
    allocator: std.mem.Allocator,
    parsed: *const module.Module,

    fn validateModule(self: *Validator) !void {
        try self.validateTypes();
        try self.validateImports();
        try self.validateExports();
        try self.validateStart();
        try self.validateLimits();
        try self.validateElements();
        try self.validateData();
        try self.validateFunctions();
    }

    fn validateTypes(self: *Validator) !void {
        for (self.parsed.function_types.items) |function_type| {
            if (function_type.results.len > 1) return error.UnsupportedMultiValueResult;
        }
    }

    fn validateImports(self: *Validator) !void {
        for (self.parsed.imports.items) |import| {
            switch (import.kind) {
                .function => {
                    const type_index = import.type_index orelse return error.MissingImportTypeIndex;
                    const function_type = try self.functionType(type_index);
                    if (function_type.params.len > 7) return error.UnsupportedImportArity;
                    if (function_type.results.len > 1) return error.UnsupportedMultiValueResult;
                    if (function_type.results.len == 1 and function_type.results[0] != .i32) {
                        return error.UnsupportedImportResultType;
                    }
                    for (function_type.params) |param| {
                        switch (param) {
                            .i32, .i64 => {},
                            else => return error.UnsupportedImportParamType,
                        }
                    }
                },
                .memory => return error.ImportedMemoriesUnsupported,
                .table => return error.ImportedTablesUnsupported,
                .global => return error.ImportedGlobalsUnsupported,
            }
        }
    }

    fn validateExports(self: *Validator) !void {
        for (self.parsed.exports.items) |exported| {
            switch (exported.kind) {
                .function => _ = try self.functionTypeIndex(exported.index),
                .memory => {
                    const actual = std.math.cast(usize, exported.index) orelse return error.InvalidMemoryIndex;
                    if (actual >= self.parsed.memories.items.len) return error.InvalidMemoryIndex;
                },
                .table => {
                    const actual = std.math.cast(usize, exported.index) orelse return error.InvalidTableIndex;
                    if (actual >= self.parsed.tables.items.len) return error.InvalidTableIndex;
                },
                .global => {
                    const actual = std.math.cast(usize, exported.index) orelse return error.InvalidGlobalIndex;
                    if (actual >= self.parsed.globals.items.len) return error.InvalidGlobalIndex;
                },
            }
        }
    }

    fn validateStart(self: *Validator) !void {
        const start_index = self.parsed.start_function_index orelse return;
        const function_type = try self.functionType(try self.functionTypeIndex(start_index));

        if (function_type.params.len != 0) return error.InvalidStartFunctionType;
        if (function_type.results.len != 0) return error.InvalidStartFunctionType;
    }

    fn validateLimits(self: *Validator) !void {
        if (self.parsed.memories.items.len > 1) return error.MultipleMemoriesUnsupported;

        for (self.parsed.memories.items) |memory| {
            try validateLimitRange(memory.limits);
        }
        for (self.parsed.tables.items) |table| {
            try validateLimitRange(table.limits);
        }
    }

    fn validateElements(self: *Validator) !void {
        for (self.parsed.element_segments.items) |segment| {
            if (!segment.passive) {
                const table_index = std.math.cast(usize, segment.table_index) orelse return error.InvalidTableIndex;
                if (table_index >= self.parsed.tables.items.len) return error.InvalidTableIndex;

                const table_min = std.math.cast(usize, self.parsed.tables.items[table_index].limits.min) orelse {
                    return error.TableTooLarge;
                };
                const offset = std.math.cast(usize, segment.offset) orelse return error.InvalidTableElementIndex;
                const end = try std.math.add(usize, offset, segment.function_indices.len);
                if (end > table_min) return error.InvalidTableElementIndex;
            }

            for (segment.function_indices) |function_index| {
                _ = try self.functionTypeIndex(function_index);
            }
        }
    }

    fn validateData(self: *Validator) !void {
        for (self.parsed.data_segments.items) |segment| {
            if (segment.passive) continue;

            const memory_index = std.math.cast(usize, segment.memory_index) orelse return error.InvalidMemoryIndex;
            if (memory_index >= self.parsed.memories.items.len) return error.InvalidMemoryIndex;

            const min_pages = std.math.cast(usize, self.parsed.memories.items[memory_index].limits.min) orelse {
                return error.MemorySizeTooLarge;
            };
            const min_bytes = try std.math.mul(usize, min_pages, wasm_page_size);
            const offset = std.math.cast(usize, segment.offset) orelse return error.InvalidDataSegmentOffset;
            const end = try std.math.add(usize, offset, segment.bytes.len);
            if (end > min_bytes) return error.DataSegmentOutOfBounds;
        }
    }

    fn validateDataIndex(self: *const Validator, data_index: u32) !void {
        const actual = std.math.cast(usize, data_index) orelse return error.InvalidDataSegmentIndex;
        if (actual >= self.parsed.data_segments.items.len) return error.InvalidDataSegmentIndex;
    }

    fn validateElementIndex(self: *const Validator, element_index: u32) !void {
        const actual = std.math.cast(usize, element_index) orelse return error.InvalidElementSegmentIndex;
        if (actual >= self.parsed.element_segments.items.len) return error.InvalidElementSegmentIndex;
    }

    fn validateFunctions(self: *Validator) !void {
        for (self.parsed.functions.items, 0..) |function, defined_index| {
            const type_index = function.type_index;
            const function_type = try self.functionType(type_index);
            const function_index = try std.math.add(usize, self.importedFunctionCount(), defined_index);

            var local_types: std.ArrayList(module.ValueType) = .empty;
            defer local_types.deinit(self.allocator);
            try local_types.appendSlice(self.allocator, function_type.params);

            var reader = binary.Reader.init(function.body);
            try readLocalTypes(self.allocator, &reader, &local_types);

            var expression = ExpressionValidator{
                .allocator = self.allocator,
                .validator = self,
                .function_index = function_index,
                .function_type = function_type,
                .locals = local_types.items,
            };
            try expression.validateFunction(&reader);
        }
    }

    fn importedFunctionCount(self: *const Validator) usize {
        var count: usize = 0;
        for (self.parsed.imports.items) |import| {
            if (import.kind == .function) count += 1;
        }
        return count;
    }

    fn functionCount(self: *const Validator) usize {
        return self.importedFunctionCount() + self.parsed.functions.items.len;
    }

    fn functionType(self: *const Validator, type_index: u32) !module.FunctionType {
        const actual = std.math.cast(usize, type_index) orelse return error.InvalidFunctionTypeIndex;
        if (actual >= self.parsed.function_types.items.len) return error.InvalidFunctionTypeIndex;
        return self.parsed.function_types.items[actual];
    }

    fn functionTypeIndex(self: *const Validator, function_index: u32) !u32 {
        const actual = std.math.cast(usize, function_index) orelse return error.InvalidFunctionIndex;
        const imported_count = self.importedFunctionCount();

        if (actual < imported_count) {
            var current: usize = 0;
            for (self.parsed.imports.items) |import| {
                if (import.kind != .function) continue;

                if (current == actual) {
                    return import.type_index orelse return error.MissingImportTypeIndex;
                }
                current += 1;
            }

            return error.InvalidFunctionIndex;
        }

        const defined_index = actual - imported_count;
        if (defined_index >= self.parsed.functions.items.len) return error.InvalidFunctionIndex;

        return self.parsed.functions.items[defined_index].type_index;
    }

    fn globalType(self: *const Validator, global_index: u32) !module.GlobalType {
        const actual = std.math.cast(usize, global_index) orelse return error.InvalidGlobalIndex;
        if (actual >= self.parsed.globals.items.len) return error.InvalidGlobalIndex;

        return self.parsed.globals.items[actual].global_type;
    }

    fn hasMemory(self: *const Validator) bool {
        return self.parsed.memories.items.len == 1;
    }

    fn hasTable(self: *const Validator, table_index: u32) bool {
        const actual = std.math.cast(usize, table_index) orelse return false;
        return actual < self.parsed.tables.items.len;
    }
};

const ExpressionValidator = struct {
    allocator: std.mem.Allocator,
    validator: *Validator,
    function_index: usize,
    function_type: module.FunctionType,
    locals: []const module.ValueType,
    stack: std.ArrayList(module.ValueType) = .empty,
    labels: std.ArrayList(BlockType) = .empty,
    frame_base: usize = 0,
    flow: ?Flow = null,

    fn validateFunction(self: *ExpressionValidator, reader: *binary.Reader) !void {
        defer self.stack.deinit(self.allocator);
        defer self.labels.deinit(self.allocator);

        if (reader.offset >= reader.bytes.len) return error.MissingFunctionEnd;
        if (reader.bytes[reader.bytes.len - 1] != 0x0b) return error.MissingFunctionEnd;

        const expression_end = reader.bytes.len - 1;
        try self.validateRange(reader, expression_end);
        if (reader.offset != expression_end) return error.TrailingFunctionBytes;

        reader.offset += 1;

        if (self.flow) |flow| {
            switch (flow) {
                .returned, .trapped => {
                    self.stack.items.len = 0;
                    try self.pushResults(self.function_type.results);
                },
                .branch => return error.InvalidFunctionBranch,
            }
        }

        try self.requireStackResults(0, self.function_type.results);
        if (!reader.isAtEnd()) return error.TrailingFunctionBytes;
    }

    fn validateRange(self: *ExpressionValidator, reader: *binary.Reader, limit: usize) anyerror!void {
        const saved_base = self.frame_base;
        self.frame_base = self.stack.items.len;
        defer self.frame_base = saved_base;

        while (reader.offset < limit) {
            const opcode = try reader.readByte();
            try self.validateInstruction(reader, limit, opcode);
        }
    }

    fn validateInstruction(
        self: *ExpressionValidator,
        reader: *binary.Reader,
        limit: usize,
        opcode: u8,
    ) anyerror!void {
        _ = limit;

        switch (opcode) {
            0x00 => self.markFlow(.trapped),
            0x01 => {},
            0x02 => try self.validateBlock(reader, .block),
            0x03 => try self.validateBlock(reader, .loop),
            0x04 => try self.validateIf(reader),
            0x05 => return error.UnexpectedElse,
            0x0b => return error.UnexpectedEnd,
            0x0c => {
                const label_index = try reader.readVarU32();
                const target_label = try self.label(label_index);
                try self.popBlockResult(target_label);
                self.markFlow(.{ .branch = label_index });
            },
            0x0d => {
                const label_index = try reader.readVarU32();
                const target_label = try self.label(label_index);
                try self.pop(.i32);
                try self.popBlockResult(target_label);
                try self.pushBlockResult(target_label);
            },
            0x0e => try self.validateBranchTable(reader),
            0x0f => {
                try self.popResults(self.function_type.results);
                self.markFlow(.returned);
            },
            0x10 => try self.validateCall(reader),
            0x11 => try self.validateCallIndirect(reader),
            0x1a => _ = try self.popAny(),
            0x1b => {
                try self.pop(.i32);
                const rhs = try self.popAny();
                const lhs = try self.popAny();
                if (lhs != rhs) return error.SelectTypeMismatch;
                try self.push(lhs);
            },
            0x1c => {
                const selected = try readSelectTypeVector(reader);
                try self.pop(.i32);
                try self.pop(selected);
                try self.pop(selected);
                try self.push(selected);
            },
            0x20 => {
                const local_index = try reader.readVarU32();
                try self.push(try self.localType(local_index));
            },
            0x21 => {
                const local_index = try reader.readVarU32();
                try self.pop(try self.localType(local_index));
            },
            0x22 => {
                const local_index = try reader.readVarU32();
                const value_type = try self.localType(local_index);
                try self.pop(value_type);
                try self.push(value_type);
            },
            0x23 => {
                const global_index = try reader.readVarU32();
                const global_type = try self.validator.globalType(global_index);
                try self.push(global_type.value_type);
            },
            0x24 => {
                const global_index = try reader.readVarU32();
                const global_type = try self.validator.globalType(global_index);
                if (!global_type.mutable) return error.ImmutableGlobal;
                try self.pop(global_type.value_type);
            },
            0x28 => try self.validateLoad(reader, .i32),
            0x29 => try self.validateLoad(reader, .i64),
            0x2a => try self.validateLoad(reader, .f32),
            0x2b => try self.validateLoad(reader, .f64),
            0x2c...0x2f => try self.validateLoad(reader, .i32),
            0x30...0x35 => try self.validateLoad(reader, .i64),
            0x36 => try self.validateStore(reader, .i32),
            0x37 => try self.validateStore(reader, .i64),
            0x38 => try self.validateStore(reader, .f32),
            0x39 => try self.validateStore(reader, .f64),
            0x3a, 0x3b => try self.validateStore(reader, .i32),
            0x3c...0x3e => try self.validateStore(reader, .i64),
            0x3f => {
                try self.readMemoryIndex(reader);
                try self.push(.i32);
            },
            0x40 => {
                try self.readMemoryIndex(reader);
                try self.pop(.i32);
                try self.push(.i32);
            },
            0x41 => {
                _ = try reader.readVarI32();
                try self.push(.i32);
            },
            0x42 => {
                _ = try reader.readVarI64();
                try self.push(.i64);
            },
            0x43 => {
                _ = try reader.readBytes(4);
                try self.push(.f32);
            },
            0x44 => {
                _ = try reader.readBytes(8);
                try self.push(.f64);
            },
            0x45 => {
                try self.pop(.i32);
                try self.push(.i32);
            },
            0x46...0x4f => try self.validateBinary(.i32, .i32),
            0x50 => {
                try self.pop(.i64);
                try self.push(.i32);
            },
            0x51...0x5a => try self.validateBinary(.i64, .i32),
            0x5b...0x60 => try self.validateBinary(.f32, .i32),
            0x61...0x66 => try self.validateBinary(.f64, .i32),
            0x67...0x69 => try self.validateUnary(.i32, .i32),
            0x6a...0x78 => try self.validateBinary(.i32, .i32),
            0x79...0x7b => try self.validateUnary(.i64, .i64),
            0x7c...0x8a => try self.validateBinary(.i64, .i64),
            0x8b...0x91 => try self.validateUnary(.f32, .f32),
            0x92...0x98 => try self.validateBinary(.f32, .f32),
            0x99...0x9f => try self.validateUnary(.f64, .f64),
            0xa0...0xa6 => try self.validateBinary(.f64, .f64),
            0xa7 => try self.validateUnary(.i64, .i32),
            0xa8, 0xa9 => try self.validateUnary(.f32, .i32),
            0xaa, 0xab => try self.validateUnary(.f64, .i32),
            0xac, 0xad => try self.validateUnary(.i32, .i64),
            0xae, 0xaf => try self.validateUnary(.f32, .i64),
            0xb0, 0xb1 => try self.validateUnary(.f64, .i64),
            0xb2, 0xb3 => try self.validateUnary(.i32, .f32),
            0xb4, 0xb5 => try self.validateUnary(.i64, .f32),
            0xb6 => try self.validateUnary(.f64, .f32),
            0xb7, 0xb8 => try self.validateUnary(.i32, .f64),
            0xb9, 0xba => try self.validateUnary(.i64, .f64),
            0xbb => try self.validateUnary(.f32, .f64),
            0xbc => try self.validateUnary(.f32, .i32),
            0xbd => try self.validateUnary(.f64, .i64),
            0xbe => try self.validateUnary(.i32, .f32),
            0xbf => try self.validateUnary(.i64, .f64),
            0xc0, 0xc1 => try self.validateUnary(.i32, .i32),
            0xc2...0xc4 => try self.validateUnary(.i64, .i64),
            0xfc => try self.validatePrefixedInstruction(reader),
            0xfd => try self.validateSimdInstruction(reader),
            0xd0 => try self.validateRefNull(reader),
            0xd2 => try self.validateRefFunc(reader),
            else => return error.UnsupportedWasmOpcode,
        }
    }

    fn validatePrefixedInstruction(self: *ExpressionValidator, reader: *binary.Reader) !void {
        const subopcode = try reader.readVarU32();

        switch (subopcode) {
            0x00, 0x01 => try self.validateUnary(.f32, .i32),
            0x02, 0x03 => try self.validateUnary(.f64, .i32),
            0x04, 0x05 => try self.validateUnary(.f32, .i64),
            0x06, 0x07 => try self.validateUnary(.f64, .i64),
            0x08 => try self.validateMemoryInit(reader),
            0x09 => try self.validateDataDrop(reader),
            0x0a => try self.validateMemoryCopy(reader),
            0x0b => try self.validateMemoryFill(reader),
            0x0c => try self.validateTableInit(reader),
            0x0d => try self.validateElementDrop(reader),
            0x0e => try self.validateTableCopy(reader),
            0x0f => try self.validateTableGrow(reader),
            0x10 => try self.validateTableSize(reader),
            0x11 => try self.validateTableFill(reader),
            else => return error.UnsupportedWasmOpcode,
        }
    }

    fn validateSimdInstruction(self: *ExpressionValidator, reader: *binary.Reader) !void {
        const subopcode = try reader.readVarU32();

        switch (subopcode) {
            0x00 => try self.validateLoad(reader, .v128),
            0x0b => try self.validateStore(reader, .v128),
            0x0c => {
                _ = try reader.readBytes(16);
                try self.push(.v128);
            },
            0x11 => try self.validateUnary(.i32, .v128),
            0x1b => {
                const lane = try reader.readByte();
                if (lane >= 4) return error.InvalidSimdLane;
                try self.validateUnary(.v128, .i32);
            },
            0xae => try self.validateBinary(.v128, .v128),
            else => return error.UnsupportedWasmOpcode,
        }
    }

    fn validateMemoryCopy(self: *ExpressionValidator, reader: *binary.Reader) !void {
        const destination_memory_index = try reader.readByte();
        const source_memory_index = try reader.readByte();
        if (destination_memory_index != 0 or source_memory_index != 0) return error.UnsupportedMemoryIndex;
        if (!self.validator.hasMemory()) return error.MissingMemory;

        try self.pop(.i32);
        try self.pop(.i32);
        try self.pop(.i32);
    }

    fn validateMemoryInit(self: *ExpressionValidator, reader: *binary.Reader) !void {
        const data_index = try reader.readVarU32();
        const memory_index = try reader.readByte();
        if (memory_index != 0) return error.UnsupportedMemoryIndex;
        if (!self.validator.hasMemory()) return error.MissingMemory;
        try self.validator.validateDataIndex(data_index);

        try self.pop(.i32);
        try self.pop(.i32);
        try self.pop(.i32);
    }

    fn validateDataDrop(self: *ExpressionValidator, reader: *binary.Reader) !void {
        const data_index = try reader.readVarU32();
        try self.validator.validateDataIndex(data_index);
    }

    fn validateTableInit(self: *ExpressionValidator, reader: *binary.Reader) !void {
        const element_index = try reader.readVarU32();
        const table_index = try reader.readVarU32();
        try self.validator.validateElementIndex(element_index);
        if (!self.validator.hasTable(table_index)) return error.MissingTable;

        try self.pop(.i32);
        try self.pop(.i32);
        try self.pop(.i32);
    }

    fn validateElementDrop(self: *ExpressionValidator, reader: *binary.Reader) !void {
        const element_index = try reader.readVarU32();
        try self.validator.validateElementIndex(element_index);
    }

    fn validateTableCopy(self: *ExpressionValidator, reader: *binary.Reader) !void {
        const destination_table_index = try reader.readVarU32();
        const source_table_index = try reader.readVarU32();
        if (!self.validator.hasTable(destination_table_index)) return error.MissingTable;
        if (!self.validator.hasTable(source_table_index)) return error.MissingTable;

        try self.pop(.i32);
        try self.pop(.i32);
        try self.pop(.i32);
    }

    fn validateTableGrow(self: *ExpressionValidator, reader: *binary.Reader) !void {
        const table_index = try reader.readVarU32();
        if (!self.validator.hasTable(table_index)) return error.MissingTable;

        try self.pop(.i32);
        try self.pop(.funcref);
        try self.push(.i32);
    }

    fn validateTableSize(self: *ExpressionValidator, reader: *binary.Reader) !void {
        const table_index = try reader.readVarU32();
        if (!self.validator.hasTable(table_index)) return error.MissingTable;

        try self.push(.i32);
    }

    fn validateTableFill(self: *ExpressionValidator, reader: *binary.Reader) !void {
        const table_index = try reader.readVarU32();
        if (!self.validator.hasTable(table_index)) return error.MissingTable;

        try self.pop(.i32);
        try self.pop(.funcref);
        try self.pop(.i32);
    }

    fn validateRefNull(self: *ExpressionValidator, reader: *binary.Reader) !void {
        const heap_type = try reader.readByte();
        if (heap_type != 0x70) return error.UnsupportedReferenceType;
        try self.push(.funcref);
    }

    fn validateRefFunc(self: *ExpressionValidator, reader: *binary.Reader) !void {
        const function_index = try reader.readVarU32();
        _ = try self.validator.functionTypeIndex(function_index);
        try self.push(.funcref);
    }

    fn validateMemoryFill(self: *ExpressionValidator, reader: *binary.Reader) !void {
        const memory_index = try reader.readByte();
        if (memory_index != 0) return error.UnsupportedMemoryIndex;
        if (!self.validator.hasMemory()) return error.MissingMemory;

        try self.pop(.i32);
        try self.pop(.i32);
        try self.pop(.i32);
    }

    fn validateBlock(self: *ExpressionValidator, reader: *binary.Reader, kind: BlockKind) anyerror!void {
        const block_type = try readBlockType(reader);
        const body_start = reader.offset;
        const boundary = try findBlockBoundary(reader.bytes, body_start);
        const start_height = self.stack.items.len;
        const outer_flow = self.flow;
        self.flow = null;

        const label_type = switch (kind) {
            .block => block_type,
            .loop => BlockType{},
        };
        try self.labels.append(self.allocator, label_type);
        try self.validateRange(reader, boundary.end_offset);
        _ = self.labels.pop();

        const inner_flow = self.flow;
        self.flow = outer_flow;
        self.stack.items.len = start_height;
        try self.pushBlockResult(block_type);

        reader.offset = boundary.end_offset + 1;

        if (inner_flow) |flow| {
            switch (flow) {
                .branch => |label_index| {
                    if (label_index == 0) {
                        return;
                    }
                    self.markFlow(.{ .branch = label_index - 1 });
                },
                .returned => self.markFlow(.returned),
                .trapped => self.markFlow(.trapped),
            }
        }
    }

    fn validateIf(self: *ExpressionValidator, reader: *binary.Reader) anyerror!void {
        try self.pop(.i32);

        const block_type = try readBlockType(reader);
        const body_start = reader.offset;
        const boundary = try findBlockBoundary(reader.bytes, body_start);
        const start_height = self.stack.items.len;

        var then_validator = try self.branchValidator();
        defer then_validator.stack.deinit(self.allocator);
        defer then_validator.labels.deinit(self.allocator);
        then_validator.stack.items.len = start_height;
        try then_validator.labels.append(self.allocator, block_type);

        var then_reader = binary.Reader.init(reader.bytes);
        then_reader.offset = body_start;
        const then_limit = boundary.else_offset orelse boundary.end_offset;
        try then_validator.validateRange(&then_reader, then_limit);
        try then_validator.finishBranch(start_height, block_type);

        var else_validator = try self.branchValidator();
        defer else_validator.stack.deinit(self.allocator);
        defer else_validator.labels.deinit(self.allocator);
        else_validator.stack.items.len = start_height;

        if (boundary.else_offset) |else_offset| {
            try else_validator.labels.append(self.allocator, block_type);
            var else_reader = binary.Reader.init(reader.bytes);
            else_reader.offset = else_offset + 1;
            try else_validator.validateRange(&else_reader, boundary.end_offset);
        } else if (block_type.result != null) {
            return error.IfResultRequiresElse;
        }
        try else_validator.finishBranch(start_height, block_type);

        self.stack.items.len = start_height;
        try self.pushBlockResult(block_type);
        self.flow = mergeFlows(then_validator.flow, else_validator.flow);
        reader.offset = boundary.end_offset + 1;
    }

    fn validateBranchTable(self: *ExpressionValidator, reader: *binary.Reader) !void {
        const label_count = try reader.readVarU32();
        const label_count_usize = std.math.cast(usize, label_count) orelse return error.BranchTableTooLarge;
        try self.pop(.i32);

        var selected_flow: ?Flow = null;
        var selected_label: ?BlockType = null;

        for (0..label_count_usize) |_| {
            const label_index = try reader.readVarU32();
            const label_type = try self.label(label_index);
            if (selected_label) |existing| {
                if (!existing.eql(label_type)) return error.BranchTableTypeMismatch;
            } else {
                selected_label = label_type;
            }
            selected_flow = try mergeBranchTableFlow(selected_flow, label_index);
        }

        const default_index = try reader.readVarU32();
        const default_label = try self.label(default_index);
        if (selected_label) |existing| {
            if (!existing.eql(default_label)) return error.BranchTableTypeMismatch;
        } else {
            selected_label = default_label;
        }
        selected_flow = try mergeBranchTableFlow(selected_flow, default_index);

        try self.popBlockResult(selected_label.?);
        self.markFlow(selected_flow.?);
    }

    fn validateCall(self: *ExpressionValidator, reader: *binary.Reader) !void {
        const function_index = try reader.readVarU32();
        const type_index = try self.validator.functionTypeIndex(function_index);
        const function_type = try self.validator.functionType(type_index);

        try self.popResultsReversed(function_type.params);
        try self.pushResults(function_type.results);
    }

    fn validateCallIndirect(self: *ExpressionValidator, reader: *binary.Reader) !void {
        const type_index = try reader.readVarU32();
        const table_index = try reader.readVarU32();
        if (!self.validator.hasTable(table_index)) return error.InvalidTableIndex;

        const function_type = try self.validator.functionType(type_index);
        try self.pop(.i32);
        try self.popResultsReversed(function_type.params);
        try self.pushResults(function_type.results);
    }

    fn validateLoad(self: *ExpressionValidator, reader: *binary.Reader, result_type: module.ValueType) !void {
        try self.readMemoryImmediate(reader);
        try self.pop(.i32);
        try self.push(result_type);
    }

    fn validateStore(self: *ExpressionValidator, reader: *binary.Reader, value_type: module.ValueType) !void {
        try self.readMemoryImmediate(reader);
        try self.pop(value_type);
        try self.pop(.i32);
    }

    fn validateUnary(
        self: *ExpressionValidator,
        operand_type: module.ValueType,
        result_type: module.ValueType,
    ) !void {
        try self.pop(operand_type);
        try self.push(result_type);
    }

    fn validateBinary(
        self: *ExpressionValidator,
        operand_type: module.ValueType,
        result_type: module.ValueType,
    ) !void {
        try self.pop(operand_type);
        try self.pop(operand_type);
        try self.push(result_type);
    }

    fn branchValidator(self: *ExpressionValidator) !ExpressionValidator {
        var stack: std.ArrayList(module.ValueType) = .empty;
        errdefer stack.deinit(self.allocator);
        try stack.appendSlice(self.allocator, self.stack.items);

        var labels: std.ArrayList(BlockType) = .empty;
        errdefer labels.deinit(self.allocator);
        try labels.appendSlice(self.allocator, self.labels.items);

        return .{
            .allocator = self.allocator,
            .validator = self.validator,
            .function_index = self.function_index,
            .function_type = self.function_type,
            .locals = self.locals,
            .stack = stack,
            .labels = labels,
            .frame_base = self.frame_base,
            .flow = self.flow,
        };
    }

    fn finishBranch(self: *ExpressionValidator, start_height: usize, block_type: BlockType) !void {
        if (self.flow != null) {
            self.stack.items.len = start_height;
            try self.pushBlockResult(block_type);
            return;
        }

        try self.requireBlockStack(start_height, block_type);
    }

    fn readMemoryImmediate(self: *ExpressionValidator, reader: *binary.Reader) !void {
        _ = try reader.readVarU32();
        _ = try reader.readVarU32();
        if (!self.validator.hasMemory()) return error.MissingMemory;
    }

    fn readMemoryIndex(self: *ExpressionValidator, reader: *binary.Reader) !void {
        const memory_index = try reader.readByte();
        if (memory_index != 0) return error.UnsupportedMemoryIndex;
        if (!self.validator.hasMemory()) return error.MissingMemory;
    }

    fn localType(self: *ExpressionValidator, local_index: u32) !module.ValueType {
        const actual = std.math.cast(usize, local_index) orelse return error.InvalidLocalIndex;
        if (actual >= self.locals.len) return error.InvalidLocalIndex;

        return self.locals[actual];
    }

    fn label(self: *ExpressionValidator, label_index: u32) !BlockType {
        const actual = std.math.cast(usize, label_index) orelse return error.InvalidBranchLabel;
        if (actual >= self.labels.items.len) return error.InvalidBranchLabel;

        return self.labels.items[self.labels.items.len - 1 - actual];
    }

    fn markFlow(self: *ExpressionValidator, flow: Flow) void {
        self.stack.items.len = self.frame_base;
        self.flow = flow;
    }

    fn push(self: *ExpressionValidator, value_type: module.ValueType) !void {
        try self.stack.append(self.allocator, value_type);
    }

    fn pushResults(self: *ExpressionValidator, results: []const module.ValueType) !void {
        for (results) |result| {
            try self.push(result);
        }
    }

    fn pushBlockResult(self: *ExpressionValidator, block_type: BlockType) !void {
        if (block_type.result) |result| {
            try self.push(result);
        }
    }

    fn popAny(self: *ExpressionValidator) !module.ValueType {
        if (self.stack.items.len == self.frame_base and self.flow != null) {
            return .i32;
        }
        if (self.stack.items.len == 0) return error.ValidationStackUnderflow;
        if (self.stack.items.len <= self.frame_base and self.flow == null) return error.ValidationStackUnderflow;

        return self.stack.pop().?;
    }

    fn pop(self: *ExpressionValidator, expected: module.ValueType) !void {
        if (self.stack.items.len == self.frame_base and self.flow != null) {
            return;
        }

        const actual = try self.popAny();
        if (actual != expected) return error.ValidationTypeMismatch;
    }

    fn popResults(self: *ExpressionValidator, results: []const module.ValueType) !void {
        var index = results.len;
        while (index > 0) {
            index -= 1;
            try self.pop(results[index]);
        }
    }

    fn popResultsReversed(self: *ExpressionValidator, params: []const module.ValueType) !void {
        try self.popResults(params);
    }

    fn popBlockResult(self: *ExpressionValidator, block_type: BlockType) !void {
        if (block_type.result) |result| {
            try self.pop(result);
        }
    }

    fn requireBlockStack(self: *ExpressionValidator, start_height: usize, block_type: BlockType) !void {
        const expected_len = start_height + if (block_type.result == null) @as(usize, 0) else @as(usize, 1);
        if (self.stack.items.len != expected_len) return error.ValidationStackShapeMismatch;

        if (block_type.result) |result| {
            if (self.stack.items[start_height] != result) return error.ValidationTypeMismatch;
        }
    }

    fn requireStackResults(self: *ExpressionValidator, start_height: usize, results: []const module.ValueType) !void {
        if (self.stack.items.len != start_height + results.len) return error.ValidationStackShapeMismatch;

        for (results, 0..) |expected, index| {
            if (self.stack.items[start_height + index] != expected) return error.ValidationTypeMismatch;
        }
    }
};

fn validateLimitRange(limits: module.Limits) !void {
    if (limits.max) |max| {
        if (limits.min > max) return error.InvalidLimits;
    }
}

fn readLocalTypes(
    allocator: std.mem.Allocator,
    reader: *binary.Reader,
    locals: *std.ArrayList(module.ValueType),
) !void {
    const group_count = try reader.readVarU32();

    for (0..group_count) |_| {
        const count = try reader.readVarU32();
        const value_type = try valueTypeFromByte(try reader.readByte());

        for (0..count) |_| {
            try locals.append(allocator, value_type);
        }
    }
}

fn readBlockType(reader: *binary.Reader) !BlockType {
    const block_type = try reader.readByte();

    return switch (block_type) {
        0x40 => .{},
        0x70 => .{ .result = .funcref },
        0x7f => .{ .result = .i32 },
        0x7e => .{ .result = .i64 },
        0x7d => .{ .result = .f32 },
        0x7c => .{ .result = .f64 },
        0x7b => .{ .result = .v128 },
        else => error.UnsupportedBlockType,
    };
}

fn readSelectTypeVector(reader: *binary.Reader) !module.ValueType {
    const count = try reader.readVarU32();
    if (count != 1) return error.UnsupportedSelectTypeVector;

    return valueTypeFromByte(try reader.readByte());
}

fn valueTypeFromByte(value: u8) !module.ValueType {
    return switch (value) {
        0x70 => .funcref,
        0x7f => .i32,
        0x7e => .i64,
        0x7d => .f32,
        0x7c => .f64,
        0x7b => .v128,
        else => error.UnsupportedValueType,
    };
}

fn mergeFlows(lhs: ?Flow, rhs: ?Flow) ?Flow {
    if (lhs == null or rhs == null) return null;
    if (flowEqual(lhs.?, rhs.?)) return lhs.?;

    return null;
}

fn flowEqual(lhs: Flow, rhs: Flow) bool {
    return switch (lhs) {
        .branch => |lhs_label| switch (rhs) {
            .branch => |rhs_label| lhs_label == rhs_label,
            else => false,
        },
        .returned => switch (rhs) {
            .returned => true,
            else => false,
        },
        .trapped => switch (rhs) {
            .trapped => true,
            else => false,
        },
    };
}

fn mergeBranchTableFlow(current: ?Flow, label_index: u32) !Flow {
    const next = Flow{ .branch = label_index };
    if (current) |existing| {
        if (!flowEqual(existing, next)) return error.UnsupportedBranchTableTargets;
    }

    return next;
}

const BlockBoundary = struct {
    else_offset: ?usize = null,
    end_offset: usize,
};

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
                _ = try readBlockType(&reader);
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
        0x1c => _ = try readSelectTypeVector(reader),
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
        0xfc => try skipPrefixedInstructionImmediate(reader),
        0xfd => try skipSimdInstructionImmediate(reader),
        0xd0 => {
            const heap_type = try reader.readByte();
            if (heap_type != 0x70) return error.UnsupportedReferenceType;
        },
        0xd2 => _ = try reader.readVarU32(),
        else => return error.UnsupportedWasmOpcode,
    }
}

fn skipPrefixedInstructionImmediate(reader: *binary.Reader) !void {
    const subopcode = try reader.readVarU32();

    switch (subopcode) {
        0x00...0x07 => {},
        0x08 => {
            _ = try reader.readVarU32();
            const memory_index = try reader.readByte();
            if (memory_index != 0) return error.UnsupportedMemoryIndex;
        },
        0x09 => _ = try reader.readVarU32(),
        0x0a => {
            const destination_memory_index = try reader.readByte();
            const source_memory_index = try reader.readByte();
            if (destination_memory_index != 0 or source_memory_index != 0) return error.UnsupportedMemoryIndex;
        },
        0x0b => {
            const memory_index = try reader.readByte();
            if (memory_index != 0) return error.UnsupportedMemoryIndex;
        },
        0x0c, 0x0e => {
            _ = try reader.readVarU32();
            _ = try reader.readVarU32();
        },
        0x0d, 0x0f, 0x10, 0x11 => _ = try reader.readVarU32(),
        else => return error.UnsupportedWasmOpcode,
    }
}

fn skipSimdInstructionImmediate(reader: *binary.Reader) !void {
    const subopcode = try reader.readVarU32();

    switch (subopcode) {
        0x00, 0x0b => {
            _ = try reader.readVarU32();
            _ = try reader.readVarU32();
        },
        0x0c => _ = try reader.readBytes(16),
        0x11 => {},
        0x1b => {
            const lane = try reader.readByte();
            if (lane >= 4) return error.InvalidSimdLane;
        },
        0xae => {},
        else => return error.UnsupportedWasmOpcode,
    }
}

test "validator accepts supported runtime fixtures" {
    const allocator = std.testing.allocator;
    const all = [_][]const u8{
        fixtures.return_i32_seven,
        fixtures.i32_integer_ops,
        fixtures.memory_integer_ops,
        fixtures.signed_i32_const,
        fixtures.if_else_branching,
        fixtures.loop_sum_to_five,
        fixtures.early_return,
        fixtures.defined_function_call,
        fixtures.indirect_function_call,
        fixtures.i32_numeric_ops_extended,
        fixtures.i32_memory16_ops,
        fixtures.control_select_and_br_table,
        fixtures.numeric_i64_and_float_ops,
        fixtures.float_conversion_ops,
        fixtures.prefixed_numeric_and_memory_ops,
        fixtures.passive_data_memory_init,
        fixtures.passive_element_table_init,
        fixtures.table_copy_grow_size_fill,
        fixtures.simd_i32x4_memory_ops,
        fixtures.extended_i64_memory_ops,
        fixtures.globals_and_start,
        fixtures.memory_size_and_grow,
        fixtures.wasi_fd_write,
        fixtures.wasi_system_basics,
    };

    for (all) |bytes| {
        var parsed = try module.Module.parse(allocator, bytes);
        defer parsed.deinit(allocator);

        try validate(allocator, &parsed);
    }
}

test "validator rejects function result type mismatch" {
    const allocator = std.testing.allocator;
    const bytes =
        "\x00asm\x01\x00\x00\x00" ++
        "\x01\x05" ++
        "\x01\x60\x00\x01\x7f" ++
        "\x03\x02" ++
        "\x01\x00" ++
        "\x07\x07" ++
        "\x01\x03run\x00\x00" ++
        "\x0a\x06" ++
        "\x01\x04\x00\x42\x01\x0b";

    var parsed = try module.Module.parse(allocator, bytes);
    defer parsed.deinit(allocator);

    try std.testing.expectError(error.ValidationTypeMismatch, validate(allocator, &parsed));
}

test "validator rejects invalid export index" {
    const allocator = std.testing.allocator;
    const bytes =
        "\x00asm\x01\x00\x00\x00" ++
        "\x01\x05" ++
        "\x01\x60\x00\x01\x7f" ++
        "\x03\x02" ++
        "\x01\x00" ++
        "\x07\x07" ++
        "\x01\x03run\x00\x63" ++
        "\x0a\x06" ++
        "\x01\x04\x00\x41\x07\x0b";

    var parsed = try module.Module.parse(allocator, bytes);
    defer parsed.deinit(allocator);

    try std.testing.expectError(error.InvalidFunctionIndex, validate(allocator, &parsed));
}

test "validator rejects stack underflow" {
    const allocator = std.testing.allocator;
    const bytes =
        "\x00asm\x01\x00\x00\x00" ++
        "\x01\x05" ++
        "\x01\x60\x00\x01\x7f" ++
        "\x03\x02" ++
        "\x01\x00" ++
        "\x07\x07" ++
        "\x01\x03run\x00\x00" ++
        "\x0a\x05" ++
        "\x01\x03\x00\x6a\x0b";

    var parsed = try module.Module.parse(allocator, bytes);
    defer parsed.deinit(allocator);

    try std.testing.expectError(error.ValidationStackUnderflow, validate(allocator, &parsed));
}
