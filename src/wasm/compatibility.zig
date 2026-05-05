const std = @import("std");
const binary = @import("binary.zig");
const imports = @import("imports.zig");
const module = @import("module.zig");
const runtime = @import("runtime.zig");
const scope = @import("../scope.zig");

const wasm_page_size: usize = 64 * 1024;
const supported_section_count = 13;

pub const Target = struct {
    default_memory_bytes: usize = scope.default_wasm_memory_bytes,
    max_memory_bytes: usize = scope.max_wasm_memory_bytes,
};

pub const Options = struct {
    initial_memory_bytes: ?usize = null,
    export_name: ?[]const u8 = null,
};

pub const ImportDetail = struct {
    kind: module.ImportKind,
    module_name: []const u8,
    name: []const u8,
    supported: bool,
};

pub const WasiRequirement = struct {
    requirement: []const u8,
    module_name: []const u8,
    name: []const u8,
    supported: bool,
};

pub const MemoryDetail = struct {
    index: usize,
    min_pages: u32,
    max_pages: ?u32,
};

pub const ExportDetail = struct {
    kind: module.ExportKind,
    name: []const u8,
    index: u32,
};

pub const RawSectionReport = struct {
    counts: [256]usize = [_]usize{0} ** 256,
    total: usize = 0,
    unsupported_count: usize = 0,
    parse_error: ?anyerror = null,

    pub fn supported(self: RawSectionReport) bool {
        return self.parse_error == null and self.unsupported_count == 0;
    }
};

pub const RawMemoryEncodingReport = struct {
    declarations: usize = 0,
    imported_declarations: usize = 0,
    defined_declarations: usize = 0,
    limits32_supported: usize = 0,
    unsupported_flags: usize = 0,
    shared: usize = 0,
    memory64: usize = 0,
    parse_error: ?anyerror = null,

    pub fn supported(self: RawMemoryEncodingReport) bool {
        return self.parse_error == null and
            self.unsupported_flags == 0 and
            self.shared == 0 and
            self.memory64 == 0;
    }
};

pub const ImportReport = struct {
    details: std.ArrayList(ImportDetail) = .empty,
    function_imports: usize = 0,
    unsupported_function_imports: usize = 0,
    unsupported_non_function_imports: usize = 0,

    fn deinit(self: *ImportReport, allocator: std.mem.Allocator) void {
        self.details.deinit(allocator);
        self.* = undefined;
    }

    pub fn supported(self: ImportReport) bool {
        return self.unsupported_function_imports == 0 and
            self.unsupported_non_function_imports == 0;
    }
};

pub const WasiRequirementReport = struct {
    details: std.ArrayList(WasiRequirement) = .empty,
    unsupported_count: usize = 0,

    fn deinit(self: *WasiRequirementReport, allocator: std.mem.Allocator) void {
        self.details.deinit(allocator);
        self.* = undefined;
    }

    pub fn supported(self: WasiRequirementReport) bool {
        return self.unsupported_count == 0;
    }
};

pub const ResourceReport = struct {
    imported_memories: usize = 0,
    memory_min_bytes: ?usize = null,
    memory_min_error: ?anyerror = null,
    requested_initial_memory_bytes: usize = 0,
    max_runtime_memory_bytes: usize = 0,
    requested_exceeds_max: bool = false,
    requested_below_min: bool = false,
    multiple_memories_unsupported: bool = false,
    memories: std.ArrayList(MemoryDetail) = .empty,

    fn deinit(self: *ResourceReport, allocator: std.mem.Allocator) void {
        self.memories.deinit(allocator);
        self.* = undefined;
    }

    pub fn supported(self: ResourceReport) bool {
        return self.imported_memories == 0 and
            self.memory_min_error == null and
            !self.requested_exceeds_max and
            !self.requested_below_min and
            !self.multiple_memories_unsupported;
    }
};

pub const UnsupportedOpcode = struct {
    function_index: usize,
    body_offset: usize,
    opcode: u8,
    prefixed: bool = false,
    subopcode: u32 = 0,
};

pub const OpcodeReport = struct {
    total: usize = 0,
    memory_instruction_count: usize = 0,
    missing_memory_count: usize = 0,
    unsupported_memory_index_count: usize = 0,
    scan_error_count: usize = 0,
    unsupported: std.ArrayList(UnsupportedOpcode) = .empty,

    fn deinit(self: *OpcodeReport, allocator: std.mem.Allocator) void {
        self.unsupported.deinit(allocator);
        self.* = undefined;
    }

    pub fn supported(self: OpcodeReport) bool {
        return self.unsupported.items.len == 0 and
            self.missing_memory_count == 0 and
            self.unsupported_memory_index_count == 0 and
            self.scan_error_count == 0;
    }
};

pub const ExportReport = struct {
    details: std.ArrayList(ExportDetail) = .empty,
    requested_export: ?[]const u8 = null,
    has_requested_export: bool = true,

    fn deinit(self: *ExportReport, allocator: std.mem.Allocator) void {
        self.details.deinit(allocator);
        self.* = undefined;
    }

    pub fn supported(self: ExportReport) bool {
        return self.has_requested_export;
    }
};

pub const Report = struct {
    sections: RawSectionReport = .{},
    memory_encoding: RawMemoryEncodingReport = .{},
    parse_error: ?anyerror = null,
    validate_error: ?anyerror = null,
    imports: ImportReport = .{},
    wasi_requirements: WasiRequirementReport = .{},
    resources: ResourceReport = .{},
    opcodes: OpcodeReport = .{},
    exports: ExportReport = .{},

    pub fn deinit(self: *Report, allocator: std.mem.Allocator) void {
        self.imports.deinit(allocator);
        self.wasi_requirements.deinit(allocator);
        self.resources.deinit(allocator);
        self.opcodes.deinit(allocator);
        self.exports.deinit(allocator);
        self.* = undefined;
    }

    pub fn supported(self: Report) bool {
        return self.sections.supported() and
            self.memory_encoding.supported() and
            self.parse_error == null and
            self.validate_error == null and
            self.imports.supported() and
            self.wasi_requirements.supported() and
            self.resources.supported() and
            self.opcodes.supported() and
            self.exports.supported();
    }

    pub fn print(self: Report) void {
        printSectionReport(self.sections);
        printRawMemoryEncodingReport(self.memory_encoding);

        if (self.parse_error) |err| {
            std.debug.print("  parse: fail ({s})\n", .{@errorName(err)});
            return;
        }
        std.debug.print("  parse: pass\n", .{});

        if (self.validate_error) |err| {
            std.debug.print("  validate: fail ({s})\n", .{@errorName(err)});
        } else {
            std.debug.print("  validate: pass\n", .{});
        }

        printImportReport(self.imports);
        printWasiRequirementReport(self.wasi_requirements);
        printResourceReport(self.resources);
        printOpcodeReport(self.opcodes);
        printExportReport(self.exports);
    }

    pub fn printJson(self: Report) void {
        std.debug.print("{{\"supported\":{},", .{self.supported()});
        printSectionsJson(self.sections);
        std.debug.print(",", .{});
        printMemoryEncodingJson(self.memory_encoding);
        std.debug.print(",\"parse_error\":", .{});
        printJsonError(self.parse_error);
        std.debug.print(",\"validate_error\":", .{});
        printJsonError(self.validate_error);
        std.debug.print(",", .{});
        printImportsJson(self.imports);
        std.debug.print(",", .{});
        printWasiRequirementsJson(self.wasi_requirements);
        std.debug.print(",", .{});
        printResourcesJson(self.resources);
        std.debug.print(",", .{});
        printOpcodesJson(self.opcodes);
        std.debug.print(",", .{});
        printExportsJson(self.exports);
        std.debug.print("}}", .{});
    }
};

const RawSection = struct {
    id: u8,
    offset: usize,
    payload: []const u8,
};

pub fn analyzeBytes(
    allocator: std.mem.Allocator,
    bytes: []const u8,
    target: Target,
    options: Options,
) !Report {
    var report = Report{
        .sections = scanRawSections(bytes),
        .memory_encoding = scanRawMemoryEncodings(bytes),
    };
    errdefer report.deinit(allocator);

    const wasm_runtime = runtime.Runtime.init(allocator);
    var parsed = wasm_runtime.parseModule(bytes) catch |err| {
        report.parse_error = err;
        return report;
    };
    defer parsed.deinit(allocator);

    try analyzeParsed(allocator, bytes, &parsed, target, options, &report);
    return report;
}

pub fn analyzeModule(
    allocator: std.mem.Allocator,
    bytes: []const u8,
    parsed: *const module.Module,
    target: Target,
    options: Options,
) !Report {
    var report = Report{
        .sections = scanRawSections(bytes),
        .memory_encoding = scanRawMemoryEncodings(bytes),
    };
    errdefer report.deinit(allocator);

    try analyzeParsed(allocator, bytes, parsed, target, options, &report);
    return report;
}

fn analyzeParsed(
    allocator: std.mem.Allocator,
    bytes: []const u8,
    parsed: *const module.Module,
    target: Target,
    options: Options,
    report: *Report,
) !void {
    _ = bytes;

    runtime.Runtime.init(allocator).validateModule(parsed) catch |err| {
        report.validate_error = err;
    };

    try inspectImports(allocator, parsed, &report.imports, &report.wasi_requirements);
    try inspectResources(allocator, parsed, target, options, &report.resources);
    report.opcodes = try scanOpcodes(allocator, parsed);
    try inspectExports(allocator, parsed, options.export_name, &report.exports);
}

fn inspectImports(
    allocator: std.mem.Allocator,
    parsed: *const module.Module,
    import_report: *ImportReport,
    wasi_report: *WasiRequirementReport,
) !void {
    for (parsed.imports.items) |import| {
        const supported = switch (import.kind) {
            .function => imports.Resolver.resolve(import.module, import.name) != null,
            .memory, .table, .global => false,
        };

        if (import.kind == .function) {
            import_report.function_imports += 1;
            if (!supported) import_report.unsupported_function_imports += 1;
        } else {
            import_report.unsupported_non_function_imports += 1;
        }

        try import_report.details.append(allocator, .{
            .kind = import.kind,
            .module_name = import.module,
            .name = import.name,
            .supported = supported,
        });

        if (wasiRequirementName(import.module)) |requirement| {
            if (!supported) wasi_report.unsupported_count += 1;
            try wasi_report.details.append(allocator, .{
                .requirement = requirement,
                .module_name = import.module,
                .name = import.name,
                .supported = supported,
            });
        }
    }
}

fn inspectResources(
    allocator: std.mem.Allocator,
    parsed: *const module.Module,
    target: Target,
    options: Options,
    report: *ResourceReport,
) !void {
    report.max_runtime_memory_bytes = target.max_memory_bytes;

    for (parsed.imports.items) |import| {
        if (import.kind == .memory) report.imported_memories += 1;
    }

    if (parsed.memories.items.len > 1) {
        report.multiple_memories_unsupported = true;
    }

    const min_memory = minimumMemoryBytes(parsed) catch |err| {
        report.memory_min_error = err;
        report.requested_initial_memory_bytes = options.initial_memory_bytes orelse target.default_memory_bytes;
        return;
    };
    report.memory_min_bytes = min_memory;

    const requested = options.initial_memory_bytes orelse @max(target.default_memory_bytes, min_memory);
    report.requested_initial_memory_bytes = requested;
    report.requested_exceeds_max = requested > target.max_memory_bytes;
    report.requested_below_min = requested < min_memory;

    for (parsed.memories.items, 0..) |memory, index| {
        try report.memories.append(allocator, .{
            .index = index,
            .min_pages = memory.limits.min,
            .max_pages = memory.limits.max,
        });
    }
}

fn inspectExports(
    allocator: std.mem.Allocator,
    parsed: *const module.Module,
    export_name: ?[]const u8,
    report: *ExportReport,
) !void {
    report.requested_export = export_name;
    report.has_requested_export = export_name == null;

    for (parsed.exports.items) |exported| {
        if (export_name) |requested| {
            if (exported.kind == .function and std.mem.eql(u8, exported.name, requested)) {
                report.has_requested_export = true;
            }
        }

        try report.details.append(allocator, .{
            .kind = exported.kind,
            .name = exported.name,
            .index = exported.index,
        });
    }
}

fn printSectionReport(report: RawSectionReport) void {
    std.debug.print("  sections: {d}\n", .{report.total});
    if (report.parse_error) |err| {
        std.debug.print("    scan fail ({s})\n", .{@errorName(err)});
        return;
    }

    for (0..supported_section_count) |id| {
        const count = report.counts[id];
        if (count == 0) continue;

        const id_u8 = std.math.cast(u8, id) orelse unreachable;
        std.debug.print("    {s} id={d} x{d} {s}\n", .{
            sectionName(id_u8),
            id,
            count,
            if (id == 0) "ignored" else "supported",
        });
    }

    for (supported_section_count..report.counts.len) |id| {
        const count = report.counts[id];
        if (count == 0) continue;

        std.debug.print("    unknown id={d} x{d} unsupported\n", .{ id, count });
    }

    std.debug.print("  unsupported_sections: {d}\n", .{report.unsupported_count});
}

fn printRawMemoryEncodingReport(report: RawMemoryEncodingReport) void {
    std.debug.print("  memory_encoding_features: {d}\n", .{report.declarations});
    if (report.parse_error) |err| {
        std.debug.print("    scan fail ({s})\n", .{@errorName(err)});
        return;
    }

    if (report.declarations == 0) {
        std.debug.print("    none\n", .{});
    } else {
        std.debug.print("    imported_memories: {d}\n", .{report.imported_declarations});
        std.debug.print("    defined_memories: {d}\n", .{report.defined_declarations});
        std.debug.print("    limits32: {d} supported\n", .{report.limits32_supported});
        if (report.unsupported_flags != 0) {
            std.debug.print("    unsupported_limits_flags: {d} unsupported\n", .{report.unsupported_flags});
        }
        if (report.shared != 0) {
            std.debug.print("    shared_memory: {d} unsupported\n", .{report.shared});
        }
        if (report.memory64 != 0) {
            std.debug.print("    memory64: {d} unsupported\n", .{report.memory64});
        }
    }
}

fn printImportReport(report: ImportReport) void {
    std.debug.print("  imports: {d}\n", .{report.details.items.len});
    for (report.details.items) |detail| {
        switch (detail.kind) {
            .function => {
                std.debug.print("    func {s}.{s} {s}\n", .{
                    detail.module_name,
                    detail.name,
                    if (detail.supported) "supported" else "unsupported",
                });
            },
            .memory, .table, .global => {
                std.debug.print("    {s} {s}.{s} unsupported\n", .{
                    @tagName(detail.kind),
                    detail.module_name,
                    detail.name,
                });
            },
        }
    }

    std.debug.print("  function_imports: {d}\n", .{report.function_imports});
    std.debug.print("  unsupported_function_imports: {d}\n", .{report.unsupported_function_imports});
    std.debug.print("  unsupported_non_function_imports: {d}\n", .{report.unsupported_non_function_imports});
}

fn printWasiRequirementReport(report: WasiRequirementReport) void {
    std.debug.print("  wasi_requirements:\n", .{});

    for (report.details.items) |detail| {
        std.debug.print("    {s} {s}.{s} {s}\n", .{
            detail.requirement,
            detail.module_name,
            detail.name,
            if (detail.supported) "supported" else "unsupported",
        });
    }

    if (report.details.items.len == 0) {
        std.debug.print("    none\n", .{});
    }

    std.debug.print("  unsupported_wasi_requirements: {d}\n", .{report.unsupported_count});
}

fn printResourceReport(report: ResourceReport) void {
    std.debug.print("  memories: {d}\n", .{report.memories.items.len});
    std.debug.print("  imported_memories: {d}{s}\n", .{
        report.imported_memories,
        if (report.imported_memories == 0) "" else " unsupported",
    });

    if (report.multiple_memories_unsupported) {
        std.debug.print("  multiple_memories: unsupported\n", .{});
    }

    if (report.memory_min_error) |err| {
        std.debug.print("  memory_min_bytes: fail ({s})\n", .{@errorName(err)});
        return;
    }

    std.debug.print("  memory_min_bytes: {d}\n", .{report.memory_min_bytes orelse 0});
    std.debug.print("  requested_initial_memory_bytes: {d}\n", .{report.requested_initial_memory_bytes});
    std.debug.print("  max_runtime_memory_bytes: {d}\n", .{report.max_runtime_memory_bytes});

    for (report.memories.items) |memory| {
        std.debug.print("    memory[{d}] min_pages={d}", .{ memory.index, memory.min_pages });
        if (memory.max_pages) |max| {
            std.debug.print(" max_pages={d}", .{max});
        } else {
            std.debug.print(" max_pages=<runtime default>", .{});
        }
        std.debug.print("\n", .{});
    }
}

fn printOpcodeReport(report: OpcodeReport) void {
    std.debug.print("  opcodes: {d}\n", .{report.total});
    std.debug.print("    unsupported_opcodes: {d}\n", .{report.unsupported.items.len});
    for (report.unsupported.items) |unsupported| {
        if (unsupported.prefixed) {
            std.debug.print("    func={d} offset={d} opcode={d} subopcode={d} unsupported\n", .{
                unsupported.function_index,
                unsupported.body_offset,
                unsupported.opcode,
                unsupported.subopcode,
            });
        } else {
            std.debug.print("    func={d} offset={d} opcode={d} unsupported\n", .{
                unsupported.function_index,
                unsupported.body_offset,
                unsupported.opcode,
            });
        }
    }

    std.debug.print("  memory_opcode_features:\n", .{});
    std.debug.print("    memory_instructions: {d}\n", .{report.memory_instruction_count});
    std.debug.print("    missing_memory_uses: {d}{s}\n", .{
        report.missing_memory_count,
        if (report.missing_memory_count == 0) "" else " unsupported",
    });
    std.debug.print("    nonzero_memory_indices: {d}{s}\n", .{
        report.unsupported_memory_index_count,
        if (report.unsupported_memory_index_count == 0) "" else " unsupported",
    });
    if (report.scan_error_count != 0) {
        std.debug.print("    scan_errors: {d}\n", .{report.scan_error_count});
    }
}

fn printExportReport(report: ExportReport) void {
    std.debug.print("  exports: {d}\n", .{report.details.items.len});

    for (report.details.items) |exported| {
        std.debug.print("    {s} {s} index={d}\n", .{
            @tagName(exported.kind),
            exported.name,
            exported.index,
        });
    }

    if (report.requested_export) |requested| {
        std.debug.print("  requested_export: {s} {s}\n", .{
            requested,
            if (report.has_requested_export) "found" else "missing",
        });
    }
}

fn printSectionsJson(report: RawSectionReport) void {
    std.debug.print("\"sections\":{{\"total\":{d},\"unsupported_count\":{d},\"parse_error\":", .{
        report.total,
        report.unsupported_count,
    });
    printJsonError(report.parse_error);
    std.debug.print(",\"items\":[", .{});

    var first = true;
    for (0..report.counts.len) |id| {
        const count = report.counts[id];
        if (count == 0) continue;

        if (!first) std.debug.print(",", .{});
        first = false;

        const id_u8 = std.math.cast(u8, id) orelse unreachable;
        std.debug.print("{{\"id\":{d},\"name\":", .{id});
        printJsonString(sectionName(id_u8));
        std.debug.print(",\"count\":{d},\"supported\":{}}}", .{
            count,
            id < supported_section_count,
        });
    }

    std.debug.print("]}}", .{});
}

fn printMemoryEncodingJson(report: RawMemoryEncodingReport) void {
    std.debug.print(
        "\"memory_encoding\":{{\"declarations\":{d},\"imported_declarations\":{d},\"defined_declarations\":{d},\"limits32_supported\":{d},\"unsupported_flags\":{d},\"shared\":{d},\"memory64\":{d},\"parse_error\":",
        .{
            report.declarations,
            report.imported_declarations,
            report.defined_declarations,
            report.limits32_supported,
            report.unsupported_flags,
            report.shared,
            report.memory64,
        },
    );
    printJsonError(report.parse_error);
    std.debug.print("}}", .{});
}

fn printImportsJson(report: ImportReport) void {
    std.debug.print(
        "\"imports\":{{\"total\":{d},\"function_imports\":{d},\"unsupported_function_imports\":{d},\"unsupported_non_function_imports\":{d},\"items\":[",
        .{
            report.details.items.len,
            report.function_imports,
            report.unsupported_function_imports,
            report.unsupported_non_function_imports,
        },
    );

    for (report.details.items, 0..) |detail, index| {
        if (index != 0) std.debug.print(",", .{});
        std.debug.print("{{\"kind\":", .{});
        printJsonString(@tagName(detail.kind));
        std.debug.print(",\"module\":", .{});
        printJsonString(detail.module_name);
        std.debug.print(",\"name\":", .{});
        printJsonString(detail.name);
        std.debug.print(",\"supported\":{}}}", .{detail.supported});
    }

    std.debug.print("]}}", .{});
}

fn printWasiRequirementsJson(report: WasiRequirementReport) void {
    std.debug.print("\"wasi_requirements\":{{\"unsupported_count\":{d},\"items\":[", .{report.unsupported_count});

    for (report.details.items, 0..) |detail, index| {
        if (index != 0) std.debug.print(",", .{});
        std.debug.print("{{\"requirement\":", .{});
        printJsonString(detail.requirement);
        std.debug.print(",\"module\":", .{});
        printJsonString(detail.module_name);
        std.debug.print(",\"name\":", .{});
        printJsonString(detail.name);
        std.debug.print(",\"supported\":{}}}", .{detail.supported});
    }

    std.debug.print("]}}", .{});
}

fn printResourcesJson(report: ResourceReport) void {
    std.debug.print(
        "\"resources\":{{\"imported_memories\":{d},\"memory_min_bytes\":",
        .{report.imported_memories},
    );
    printJsonOptionalUsize(report.memory_min_bytes);
    std.debug.print(",\"memory_min_error\":", .{});
    printJsonError(report.memory_min_error);
    std.debug.print(
        ",\"requested_initial_memory_bytes\":{d},\"max_runtime_memory_bytes\":{d},\"requested_exceeds_max\":{},\"requested_below_min\":{},\"multiple_memories_unsupported\":{},\"memories\":[",
        .{
            report.requested_initial_memory_bytes,
            report.max_runtime_memory_bytes,
            report.requested_exceeds_max,
            report.requested_below_min,
            report.multiple_memories_unsupported,
        },
    );

    for (report.memories.items, 0..) |memory, index| {
        if (index != 0) std.debug.print(",", .{});
        std.debug.print("{{\"index\":{d},\"min_pages\":{d},\"max_pages\":", .{
            memory.index,
            memory.min_pages,
        });
        if (memory.max_pages) |max| {
            std.debug.print("{d}", .{max});
        } else {
            std.debug.print("null", .{});
        }
        std.debug.print("}}", .{});
    }

    std.debug.print("]}}", .{});
}

fn printOpcodesJson(report: OpcodeReport) void {
    std.debug.print(
        "\"opcodes\":{{\"total\":{d},\"memory_instruction_count\":{d},\"missing_memory_count\":{d},\"unsupported_memory_index_count\":{d},\"scan_error_count\":{d},\"unsupported\":[",
        .{
            report.total,
            report.memory_instruction_count,
            report.missing_memory_count,
            report.unsupported_memory_index_count,
            report.scan_error_count,
        },
    );

    for (report.unsupported.items, 0..) |unsupported, index| {
        if (index != 0) std.debug.print(",", .{});
        std.debug.print(
            "{{\"function_index\":{d},\"body_offset\":{d},\"opcode\":{d},\"prefixed\":{}",
            .{
                unsupported.function_index,
                unsupported.body_offset,
                unsupported.opcode,
                unsupported.prefixed,
            },
        );
        if (unsupported.prefixed) {
            std.debug.print(",\"subopcode\":{d}", .{unsupported.subopcode});
        }
        std.debug.print("}}", .{});
    }

    std.debug.print("]}}", .{});
}

fn printExportsJson(report: ExportReport) void {
    std.debug.print("\"exports\":{{\"requested_export\":", .{});
    if (report.requested_export) |requested| {
        printJsonString(requested);
    } else {
        std.debug.print("null", .{});
    }
    std.debug.print(",\"has_requested_export\":{},\"items\":[", .{report.has_requested_export});

    for (report.details.items, 0..) |exported, index| {
        if (index != 0) std.debug.print(",", .{});
        std.debug.print("{{\"kind\":", .{});
        printJsonString(@tagName(exported.kind));
        std.debug.print(",\"name\":", .{});
        printJsonString(exported.name);
        std.debug.print(",\"index\":{d}}}", .{exported.index});
    }

    std.debug.print("]}}", .{});
}

fn printJsonError(maybe_error: ?anyerror) void {
    if (maybe_error) |err| {
        printJsonString(@errorName(err));
    } else {
        std.debug.print("null", .{});
    }
}

fn printJsonOptionalUsize(value: ?usize) void {
    if (value) |actual| {
        std.debug.print("{d}", .{actual});
    } else {
        std.debug.print("null", .{});
    }
}

fn printJsonString(value: []const u8) void {
    std.debug.print("\"", .{});
    for (value) |char| {
        switch (char) {
            '"' => std.debug.print("\\\"", .{}),
            '\\' => std.debug.print("\\\\", .{}),
            '\n' => std.debug.print("\\n", .{}),
            '\r' => std.debug.print("\\r", .{}),
            '\t' => std.debug.print("\\t", .{}),
            else => std.debug.print("{c}", .{char}),
        }
    }
    std.debug.print("\"", .{});
}

pub fn scanRawSections(bytes: []const u8) RawSectionReport {
    var report = RawSectionReport{};
    var reader = binary.Reader.init(bytes);

    reader.readHeader() catch |err| {
        report.parse_error = err;
        return report;
    };

    while (true) {
        const section = nextRawSection(&reader) catch |err| {
            report.parse_error = err;
            return report;
        } orelse break;

        report.total += 1;
        report.counts[section.id] += 1;
        if (section.id >= supported_section_count) report.unsupported_count += 1;
    }

    return report;
}

pub fn scanRawMemoryEncodings(bytes: []const u8) RawMemoryEncodingReport {
    var report = RawMemoryEncodingReport{};
    var reader = binary.Reader.init(bytes);

    reader.readHeader() catch |err| {
        report.parse_error = err;
        return report;
    };

    while (true) {
        const section = nextRawSection(&reader) catch |err| {
            report.parse_error = err;
            return report;
        } orelse break;

        switch (section.id) {
            2 => scanImportMemoryEncodings(section.payload, &report) catch |err| {
                report.parse_error = err;
                return report;
            },
            5 => scanDefinedMemoryEncodings(section.payload, &report) catch |err| {
                report.parse_error = err;
                return report;
            },
            else => {},
        }
    }

    return report;
}

fn nextRawSection(reader: *binary.Reader) !?RawSection {
    if (reader.offset == reader.bytes.len) return null;
    if (reader.offset > reader.bytes.len) return error.InvalidSectionRange;

    const offset = reader.offset;
    const id = try reader.readByte();
    const payload_len = try reader.readVarU32();
    const payload = try reader.readBytes(payload_len);

    return .{
        .id = id,
        .offset = offset,
        .payload = payload,
    };
}

fn scanImportMemoryEncodings(payload: []const u8, report: *RawMemoryEncodingReport) !void {
    var reader = binary.Reader.init(payload);
    const count = try reader.readVarU32();

    for (0..count) |_| {
        try skipRawName(&reader);
        try skipRawName(&reader);

        const kind = try reader.readByte();
        switch (kind) {
            0 => _ = try reader.readVarU32(),
            1 => {
                _ = try reader.readByte();
                try skipRawLimits(&reader);
            },
            2 => {
                report.imported_declarations += 1;
                try scanRawMemoryLimits(&reader, report);
            },
            3 => {
                _ = try reader.readByte();
                _ = try reader.readByte();
            },
            else => return error.UnsupportedImportKind,
        }
    }
}

fn scanDefinedMemoryEncodings(payload: []const u8, report: *RawMemoryEncodingReport) !void {
    var reader = binary.Reader.init(payload);
    const count = try reader.readVarU32();

    for (0..count) |_| {
        report.defined_declarations += 1;
        try scanRawMemoryLimits(&reader, report);
    }
}

fn scanRawMemoryLimits(reader: *binary.Reader, report: *RawMemoryEncodingReport) !void {
    const flags = try reader.readVarU32();

    report.declarations += 1;
    if (flags == 0 or flags == 1) {
        report.limits32_supported += 1;
    } else {
        report.unsupported_flags += 1;
    }
    if ((flags & 0x2) != 0) report.shared += 1;
    if ((flags & 0x4) != 0) report.memory64 += 1;

    const memory64 = (flags & 0x4) != 0;
    if (memory64) {
        _ = try reader.readVarU64();
        if ((flags & 0x1) != 0) _ = try reader.readVarU64();
    } else {
        _ = try reader.readVarU32();
        if ((flags & 0x1) != 0) _ = try reader.readVarU32();
    }
}

fn skipRawLimits(reader: *binary.Reader) !void {
    const flags = try reader.readVarU32();
    _ = try reader.readVarU32();
    if ((flags & 0x1) != 0) _ = try reader.readVarU32();
}

fn skipRawName(reader: *binary.Reader) !void {
    const len = try reader.readVarU32();
    _ = try reader.readBytes(len);
}

pub fn scanOpcodes(allocator: std.mem.Allocator, parsed: *const module.Module) !OpcodeReport {
    var report = OpcodeReport{};
    errdefer report.deinit(allocator);

    const imported_count = importedFunctionCount(parsed);
    const has_memory = parsed.memories.items.len == 1;

    for (parsed.functions.items, 0..) |function, defined_index| {
        try scanFunctionOpcodes(
            allocator,
            function.body,
            imported_count + defined_index,
            has_memory,
            &report,
        );
    }

    return report;
}

fn scanFunctionOpcodes(
    allocator: std.mem.Allocator,
    body: []const u8,
    function_index: usize,
    has_memory: bool,
    report: *OpcodeReport,
) !void {
    var reader = binary.Reader.init(body);
    skipLocalDeclarations(&reader) catch {
        report.scan_error_count += 1;
        return;
    };

    while (!reader.isAtEnd()) {
        const opcode_offset = reader.offset;
        const opcode = reader.readByte() catch {
            report.scan_error_count += 1;
            return;
        };
        report.total += 1;

        const supported = skipSupportedOpcodeImmediate(
            allocator,
            &reader,
            opcode,
            function_index,
            opcode_offset,
            has_memory,
            report,
        ) catch {
            report.scan_error_count += 1;
            return;
        };

        if (!supported) {
            if (opcode != 0xfc and opcode != 0xfd) {
                try report.unsupported.append(allocator, .{
                    .function_index = function_index,
                    .body_offset = opcode_offset,
                    .opcode = opcode,
                });
            }
            return;
        }
    }
}

fn skipSupportedOpcodeImmediate(
    allocator: std.mem.Allocator,
    reader: *binary.Reader,
    opcode: u8,
    function_index: usize,
    opcode_offset: usize,
    has_memory: bool,
    report: *OpcodeReport,
) !bool {
    switch (opcode) {
        0x00, 0x01, 0x05, 0x0b, 0x0f, 0x1a, 0x1b, 0x45...0xc4 => return true,
        0x02, 0x03, 0x04 => {
            try skipBlockType(reader);
            return true;
        },
        0x0c, 0x0d, 0x10, 0x20, 0x21, 0x22, 0x23, 0x24 => {
            _ = try reader.readVarU32();
            return true;
        },
        0x0e => {
            const label_count = try reader.readVarU32();
            for (0..label_count) |_| {
                _ = try reader.readVarU32();
            }
            _ = try reader.readVarU32();
            return true;
        },
        0x11 => {
            _ = try reader.readVarU32();
            _ = try reader.readVarU32();
            return true;
        },
        0x1c => {
            try skipSelectTypeVector(reader);
            return true;
        },
        0x28...0x3e => {
            report.memory_instruction_count += 1;
            if (!has_memory) report.missing_memory_count += 1;
            _ = try reader.readVarU32();
            _ = try reader.readVarU32();
            return true;
        },
        0x3f, 0x40 => {
            report.memory_instruction_count += 1;
            if (!has_memory) report.missing_memory_count += 1;
            const memory_index = try reader.readByte();
            if (memory_index != 0) report.unsupported_memory_index_count += 1;
            return true;
        },
        0x41 => {
            _ = try reader.readVarI32();
            return true;
        },
        0x42 => {
            _ = try reader.readVarI64();
            return true;
        },
        0x43 => {
            _ = try reader.readBytes(4);
            return true;
        },
        0x44 => {
            _ = try reader.readBytes(8);
            return true;
        },
        0xfc => return try skipSupportedPrefixedOpcodeImmediate(
            allocator,
            reader,
            function_index,
            opcode_offset,
            has_memory,
            report,
        ),
        0xfd => return try skipSupportedSimdOpcodeImmediate(
            allocator,
            reader,
            function_index,
            opcode_offset,
            has_memory,
            report,
        ),
        else => return false,
    }
}

fn skipSupportedPrefixedOpcodeImmediate(
    allocator: std.mem.Allocator,
    reader: *binary.Reader,
    function_index: usize,
    opcode_offset: usize,
    has_memory: bool,
    report: *OpcodeReport,
) !bool {
    const subopcode = try reader.readVarU32();

    switch (subopcode) {
        0x00...0x07 => return true,
        0x0a => {
            report.memory_instruction_count += 1;
            if (!has_memory) report.missing_memory_count += 1;
            const destination_memory_index = try reader.readByte();
            const source_memory_index = try reader.readByte();
            if (destination_memory_index != 0 or source_memory_index != 0) {
                report.unsupported_memory_index_count += 1;
            }
            return true;
        },
        0x0b => {
            report.memory_instruction_count += 1;
            if (!has_memory) report.missing_memory_count += 1;
            const memory_index = try reader.readByte();
            if (memory_index != 0) report.unsupported_memory_index_count += 1;
            return true;
        },
        else => {
            try report.unsupported.append(allocator, .{
                .function_index = function_index,
                .body_offset = opcode_offset,
                .opcode = 0xfc,
                .prefixed = true,
                .subopcode = subopcode,
            });
            return false;
        },
    }
}

fn skipSupportedSimdOpcodeImmediate(
    allocator: std.mem.Allocator,
    reader: *binary.Reader,
    function_index: usize,
    opcode_offset: usize,
    has_memory: bool,
    report: *OpcodeReport,
) !bool {
    const subopcode = try reader.readVarU32();

    switch (subopcode) {
        0x00, 0x0b => {
            report.memory_instruction_count += 1;
            if (!has_memory) report.missing_memory_count += 1;
            _ = try reader.readVarU32();
            _ = try reader.readVarU32();
            return true;
        },
        0x0c => {
            _ = try reader.readBytes(16);
            return true;
        },
        0x11, 0xae => return true,
        0x1b => {
            const lane = try reader.readByte();
            if (lane >= 4) return error.InvalidSimdLane;
            return true;
        },
        else => {
            try report.unsupported.append(allocator, .{
                .function_index = function_index,
                .body_offset = opcode_offset,
                .opcode = 0xfd,
                .prefixed = true,
                .subopcode = subopcode,
            });
            return false;
        },
    }
}

fn skipLocalDeclarations(reader: *binary.Reader) !void {
    const group_count = try reader.readVarU32();

    for (0..group_count) |_| {
        _ = try reader.readVarU32();
        try skipValueType(reader);
    }
}

fn skipBlockType(reader: *binary.Reader) !void {
    const block_type = try reader.readByte();
    switch (block_type) {
        0x40, 0x7f, 0x7e, 0x7d, 0x7c, 0x7b => {},
        else => return error.UnsupportedBlockType,
    }
}

fn skipSelectTypeVector(reader: *binary.Reader) !void {
    const count = try reader.readVarU32();
    if (count != 1) return error.UnsupportedSelectTypeVector;
    try skipValueType(reader);
}

fn skipValueType(reader: *binary.Reader) !void {
    const value_type = try reader.readByte();
    switch (value_type) {
        0x7f, 0x7e, 0x7d, 0x7c, 0x7b => {},
        else => return error.UnsupportedValueType,
    }
}

fn importedFunctionCount(parsed: *const module.Module) usize {
    var count: usize = 0;
    for (parsed.imports.items) |import| {
        if (import.kind == .function) count += 1;
    }
    return count;
}

fn minimumMemoryBytes(parsed: *const module.Module) !usize {
    if (parsed.memories.items.len == 0) return 0;
    if (parsed.memories.items.len > 1) return error.MultipleMemoriesUnsupported;

    const pages = std.math.cast(usize, parsed.memories.items[0].limits.min) orelse {
        return error.MemorySizeTooLarge;
    };

    return try std.math.mul(usize, pages, wasm_page_size);
}

fn wasiRequirementName(module_name: []const u8) ?[]const u8 {
    if (std.mem.eql(u8, module_name, imports.wasi_module_name)) return "wasi-preview1";
    if (std.mem.eql(u8, module_name, imports.wasi_nn_module_name)) return "wasi-nn";
    if (std.mem.startsWith(u8, module_name, "wasi")) return "wasi-unknown";

    return null;
}

fn sectionName(id: u8) []const u8 {
    return switch (id) {
        0 => "custom",
        1 => "type",
        2 => "import",
        3 => "function",
        4 => "table",
        5 => "memory",
        6 => "global",
        7 => "export",
        8 => "start",
        9 => "element",
        10 => "code",
        11 => "data",
        12 => "data_count",
        else => "unknown",
    };
}

test "compatibility accepts supported wasm fixture" {
    const allocator = std.testing.allocator;
    const fixtures = @import("fixtures.zig");

    var report = try analyzeBytes(allocator, fixtures.return_i32_seven, .{}, .{ .export_name = "run" });
    defer report.deinit(allocator);

    try std.testing.expect(report.supported());
    try std.testing.expectEqual(@as(usize, 0), report.sections.unsupported_count);
    try std.testing.expectEqual(@as(usize, 0), report.imports.unsupported_function_imports);
}

test "compatibility accepts supported simd wasm fixture" {
    const allocator = std.testing.allocator;
    const fixtures = @import("fixtures.zig");

    var report = try analyzeBytes(allocator, fixtures.simd_i32x4_memory_ops, .{}, .{ .export_name = "run" });
    defer report.deinit(allocator);

    try std.testing.expect(report.supported());
    try std.testing.expectEqual(@as(usize, 2), report.opcodes.memory_instruction_count);
    try std.testing.expectEqual(@as(usize, 0), report.opcodes.unsupported.items.len);
}

test "compatibility scans unsupported wasm sections before parse" {
    const bytes =
        "\x00asm\x01\x00\x00\x00" ++
        "\x0d\x00";

    const report = scanRawSections(bytes);

    try std.testing.expect(report.parse_error == null);
    try std.testing.expectEqual(@as(usize, 1), report.total);
    try std.testing.expectEqual(@as(usize, 1), report.unsupported_count);
    try std.testing.expectEqual(@as(usize, 1), report.counts[13]);
}

test "compatibility scans unsupported memory encoding features before parse" {
    const shared_memory_bytes =
        "\x00asm\x01\x00\x00\x00" ++
        "\x05\x04" ++
        "\x01\x03\x01\x02";

    const shared = scanRawMemoryEncodings(shared_memory_bytes);
    try std.testing.expect(shared.parse_error == null);
    try std.testing.expectEqual(@as(usize, 1), shared.declarations);
    try std.testing.expectEqual(@as(usize, 1), shared.shared);
    try std.testing.expectEqual(@as(usize, 1), shared.unsupported_flags);

    const memory64_bytes =
        "\x00asm\x01\x00\x00\x00" ++
        "\x05\x04" ++
        "\x01\x05\x01\x02";

    const memory64 = scanRawMemoryEncodings(memory64_bytes);
    try std.testing.expect(memory64.parse_error == null);
    try std.testing.expectEqual(@as(usize, 1), memory64.declarations);
    try std.testing.expectEqual(@as(usize, 1), memory64.memory64);
    try std.testing.expectEqual(@as(usize, 1), memory64.unsupported_flags);
}

test "compatibility scans unsupported opcodes without executing" {
    const allocator = std.testing.allocator;

    const bytes =
        "\x00asm\x01\x00\x00\x00" ++
        "\x01\x04" ++
        "\x01\x60\x00\x00" ++
        "\x03\x02" ++
        "\x01\x00" ++
        "\x0a\x07" ++
        "\x01\x05\x00\xfd\xff\x01\x0b";

    var parsed = try module.Module.parse(allocator, bytes);
    defer parsed.deinit(allocator);

    var report = try scanOpcodes(allocator, &parsed);
    defer report.deinit(allocator);

    try std.testing.expectEqual(@as(usize, 1), report.unsupported.items.len);
    try std.testing.expectEqual(@as(u8, 0xfd), report.unsupported.items[0].opcode);
    try std.testing.expect(report.unsupported.items[0].prefixed);
    try std.testing.expectEqual(@as(u32, 255), report.unsupported.items[0].subopcode);
}
