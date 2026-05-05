const std = @import("std");

pub const AllocationStats = struct {
    alloc_calls: usize = 0,
    free_calls: usize = 0,
    resize_calls: usize = 0,
    remap_calls: usize = 0,
    allocated_bytes: usize = 0,
    freed_bytes: usize = 0,
    current_bytes: usize = 0,
    peak_bytes: usize = 0,
};

pub const AllocationDelta = struct {
    alloc_calls: usize = 0,
    free_calls: usize = 0,
    resize_calls: usize = 0,
    remap_calls: usize = 0,
    allocated_bytes: usize = 0,
    freed_bytes: usize = 0,
    net_bytes: i128 = 0,
};

pub const TimingSummary = struct {
    count: usize,
    total_ns: i128,
    avg_ns: i128,
    min_ns: i128,
    p50_ns: i128,
    p95_ns: i128,
    max_ns: i128,
};

pub const CountingAllocator = struct {
    backing: std.mem.Allocator,
    stats: AllocationStats = .{},

    pub fn init(backing: std.mem.Allocator) CountingAllocator {
        return .{
            .backing = backing,
        };
    }

    pub fn allocator(self: *CountingAllocator) std.mem.Allocator {
        return .{
            .ptr = self,
            .vtable = &vtable,
        };
    }

    pub fn snapshot(self: *const CountingAllocator) AllocationStats {
        return self.stats;
    }

    fn recordAlloc(self: *CountingAllocator, len: usize) void {
        self.stats.alloc_calls += 1;
        self.stats.allocated_bytes = addSaturating(self.stats.allocated_bytes, len);
        self.stats.current_bytes = addSaturating(self.stats.current_bytes, len);
        self.stats.peak_bytes = @max(self.stats.peak_bytes, self.stats.current_bytes);
    }

    fn recordResize(self: *CountingAllocator, old_len: usize, new_len: usize) void {
        self.stats.resize_calls += 1;
        self.recordSizeChange(old_len, new_len);
    }

    fn recordRemap(self: *CountingAllocator, old_len: usize, new_len: usize) void {
        self.stats.remap_calls += 1;
        self.recordSizeChange(old_len, new_len);
    }

    fn recordSizeChange(self: *CountingAllocator, old_len: usize, new_len: usize) void {
        if (new_len > old_len) {
            const delta = new_len - old_len;
            self.stats.allocated_bytes = addSaturating(self.stats.allocated_bytes, delta);
            self.stats.current_bytes = addSaturating(self.stats.current_bytes, delta);
            self.stats.peak_bytes = @max(self.stats.peak_bytes, self.stats.current_bytes);
        } else if (old_len > new_len) {
            const delta = old_len - new_len;
            self.stats.freed_bytes = addSaturating(self.stats.freed_bytes, delta);
            self.stats.current_bytes = subClamped(self.stats.current_bytes, delta);
        }
    }

    fn recordFree(self: *CountingAllocator, len: usize) void {
        self.stats.free_calls += 1;
        self.stats.freed_bytes = addSaturating(self.stats.freed_bytes, len);
        self.stats.current_bytes = subClamped(self.stats.current_bytes, len);
    }

    const vtable = std.mem.Allocator.VTable{
        .alloc = alloc,
        .resize = resize,
        .remap = remap,
        .free = free,
    };

    fn alloc(ctx: *anyopaque, len: usize, alignment: std.mem.Alignment, ret_addr: usize) ?[*]u8 {
        const self: *CountingAllocator = @ptrCast(@alignCast(ctx));
        const ptr = self.backing.rawAlloc(len, alignment, ret_addr) orelse return null;
        self.recordAlloc(len);
        return ptr;
    }

    fn resize(
        ctx: *anyopaque,
        memory: []u8,
        alignment: std.mem.Alignment,
        new_len: usize,
        ret_addr: usize,
    ) bool {
        const self: *CountingAllocator = @ptrCast(@alignCast(ctx));
        if (!self.backing.rawResize(memory, alignment, new_len, ret_addr)) return false;
        self.recordResize(memory.len, new_len);
        return true;
    }

    fn remap(
        ctx: *anyopaque,
        memory: []u8,
        alignment: std.mem.Alignment,
        new_len: usize,
        ret_addr: usize,
    ) ?[*]u8 {
        const self: *CountingAllocator = @ptrCast(@alignCast(ctx));
        const ptr = self.backing.rawRemap(memory, alignment, new_len, ret_addr) orelse return null;
        self.recordRemap(memory.len, new_len);
        return ptr;
    }

    fn free(ctx: *anyopaque, memory: []u8, alignment: std.mem.Alignment, ret_addr: usize) void {
        const self: *CountingAllocator = @ptrCast(@alignCast(ctx));
        self.recordFree(memory.len);
        self.backing.rawFree(memory, alignment, ret_addr);
    }
};

pub fn allocationDelta(before: AllocationStats, after: AllocationStats) AllocationDelta {
    return .{
        .alloc_calls = subClamped(after.alloc_calls, before.alloc_calls),
        .free_calls = subClamped(after.free_calls, before.free_calls),
        .resize_calls = subClamped(after.resize_calls, before.resize_calls),
        .remap_calls = subClamped(after.remap_calls, before.remap_calls),
        .allocated_bytes = subClamped(after.allocated_bytes, before.allocated_bytes),
        .freed_bytes = subClamped(after.freed_bytes, before.freed_bytes),
        .net_bytes = @as(i128, @intCast(after.current_bytes)) - @as(i128, @intCast(before.current_bytes)),
    };
}

pub fn summarizeTimings(allocator: std.mem.Allocator, samples: []const i128) !TimingSummary {
    if (samples.len == 0) return error.EmptyBenchmark;

    const sorted = try allocator.dupe(i128, samples);
    defer allocator.free(sorted);
    sortI128(sorted);

    var total: i128 = 0;
    for (samples) |sample| {
        total += sample;
    }

    return .{
        .count = samples.len,
        .total_ns = total,
        .avg_ns = @divTrunc(total, @as(i128, @intCast(samples.len))),
        .min_ns = sorted[0],
        .p50_ns = sorted[percentileIndex(samples.len, 50)],
        .p95_ns = sorted[percentileIndex(samples.len, 95)],
        .max_ns = sorted[sorted.len - 1],
    };
}

pub fn nowNs() i128 {
    return std.Io.Clock.awake.now(std.Options.debug_io).toNanoseconds();
}

pub fn nsToMs(ns: i128) f64 {
    return @as(f64, @floatFromInt(ns)) / 1_000_000.0;
}

fn percentileIndex(len: usize, percentile: usize) usize {
    const one_based = (len * percentile + 99) / 100;
    return @min(len - 1, if (one_based == 0) 0 else one_based - 1);
}

fn sortI128(values: []i128) void {
    var index: usize = 1;
    while (index < values.len) : (index += 1) {
        const value = values[index];
        var scan = index;
        while (scan > 0 and values[scan - 1] > value) : (scan -= 1) {
            values[scan] = values[scan - 1];
        }
        values[scan] = value;
    }
}

fn addSaturating(lhs: usize, rhs: usize) usize {
    return std.math.add(usize, lhs, rhs) catch std.math.maxInt(usize);
}

fn subClamped(lhs: usize, rhs: usize) usize {
    return if (rhs > lhs) 0 else lhs - rhs;
}

test "summarizeTimings reports sorted percentile stats" {
    const allocator = std.testing.allocator;
    const summary = try summarizeTimings(allocator, &.{ 30, 10, 50, 20, 40 });

    try std.testing.expectEqual(@as(usize, 5), summary.count);
    try std.testing.expectEqual(@as(i128, 150), summary.total_ns);
    try std.testing.expectEqual(@as(i128, 30), summary.avg_ns);
    try std.testing.expectEqual(@as(i128, 10), summary.min_ns);
    try std.testing.expectEqual(@as(i128, 30), summary.p50_ns);
    try std.testing.expectEqual(@as(i128, 50), summary.p95_ns);
    try std.testing.expectEqual(@as(i128, 50), summary.max_ns);
}

test "CountingAllocator tracks live and peak bytes" {
    var counter = CountingAllocator.init(std.testing.allocator);
    const allocator = counter.allocator();

    const before = counter.snapshot();
    const bytes = try allocator.alloc(u8, 128);
    const after_alloc = counter.snapshot();

    try std.testing.expectEqual(@as(usize, 1), after_alloc.alloc_calls - before.alloc_calls);
    try std.testing.expectEqual(@as(usize, 128), after_alloc.current_bytes - before.current_bytes);
    try std.testing.expect(after_alloc.peak_bytes >= 128);

    allocator.free(bytes);
    const after_free = counter.snapshot();
    const delta = allocationDelta(before, after_free);

    try std.testing.expectEqual(@as(usize, 1), delta.alloc_calls);
    try std.testing.expectEqual(@as(usize, 1), delta.free_calls);
    try std.testing.expectEqual(@as(usize, 128), delta.allocated_bytes);
    try std.testing.expectEqual(@as(usize, 128), delta.freed_bytes);
    try std.testing.expectEqual(@as(i128, 0), delta.net_bytes);
}
