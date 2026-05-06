const std = @import("std");

pub const DeviceKind = enum(u32) {
    unknown = 0,
    cpu_fallback = 1,
    integrated = 2,
    discrete = 3,
    accelerator = 4,
};

pub const Device = struct {
    kind: DeviceKind = .unknown,
    total_memory_bytes: u64 = 0,
    available_memory_bytes: u64 = 0,
    queue_count: u32 = 1,
};

pub const DeviceHandle = u32;
pub const QueueHandle = u32;
pub const BufferHandle = u32;

pub const default_max_buffer_bytes: usize = 64 * 1024 * 1024;

pub const Capabilities = struct {
    enabled: bool = false,
    devices: []const Device = &.{},

    pub fn deviceCount(self: Capabilities) usize {
        if (!self.enabled) return 0;
        return self.devices.len;
    }

    pub fn device(self: Capabilities, index: u32) ?Device {
        const actual = std.math.cast(usize, index) orelse return null;
        if (actual >= self.deviceCount()) return null;
        return self.devices[actual];
    }
};

pub const mock_edge_devices = [_]Device{.{
    .kind = .integrated,
    .total_memory_bytes = 512 * 1024 * 1024,
    .available_memory_bytes = 384 * 1024 * 1024,
    .queue_count = 1,
}};

pub fn mockEdgeCapabilities() Capabilities {
    return .{
        .enabled = true,
        .devices = &mock_edge_devices,
    };
}

const DeviceBinding = struct {
    device_index: u32,
};

const QueueBinding = struct {
    device_handle: DeviceHandle,
    queue_index: u32,
};

const Buffer = struct {
    device_handle: DeviceHandle,
    bytes: []u8,

    fn deinit(self: *Buffer, allocator: std.mem.Allocator) void {
        if (self.bytes.len != 0) allocator.free(self.bytes);
        self.* = undefined;
    }
};

pub const Runtime = struct {
    devices: std.ArrayList(DeviceBinding) = .empty,
    queues: std.ArrayList(QueueBinding) = .empty,
    buffers: std.ArrayList(Buffer) = .empty,
    max_buffer_bytes: usize = default_max_buffer_bytes,

    pub fn deinit(self: *Runtime, allocator: std.mem.Allocator) void {
        for (self.buffers.items) |*buffer| {
            buffer.deinit(allocator);
        }
        self.buffers.deinit(allocator);
        self.queues.deinit(allocator);
        self.devices.deinit(allocator);
        self.* = .{};
    }

    pub fn openDevice(
        self: *Runtime,
        allocator: std.mem.Allocator,
        caps: Capabilities,
        device_index: u32,
    ) !DeviceHandle {
        _ = caps.device(device_index) orelse return error.InvalidDevice;
        const handle = try nextHandle(self.devices.items.len);
        try self.devices.append(allocator, .{ .device_index = device_index });
        return handle;
    }

    pub fn defaultQueue(
        self: *Runtime,
        allocator: std.mem.Allocator,
        caps: Capabilities,
        device_handle: DeviceHandle,
    ) !QueueHandle {
        const binding = self.deviceBinding(device_handle) orelse return error.InvalidDeviceHandle;
        const device = caps.device(binding.device_index) orelse return error.InvalidDevice;
        if (device.queue_count == 0) return error.QueueUnavailable;

        const handle = try nextHandle(self.queues.items.len);
        try self.queues.append(allocator, .{
            .device_handle = device_handle,
            .queue_index = 0,
        });
        return handle;
    }

    pub fn createBuffer(
        self: *Runtime,
        allocator: std.mem.Allocator,
        caps: Capabilities,
        device_handle: DeviceHandle,
        size: u32,
    ) !BufferHandle {
        const binding = self.deviceBinding(device_handle) orelse return error.InvalidDeviceHandle;
        const device = caps.device(binding.device_index) orelse return error.InvalidDevice;
        const actual_size = std.math.cast(usize, size) orelse return error.InvalidSize;
        const actual_size_u64 = std.math.cast(u64, actual_size) orelse return error.BufferTooLarge;
        if (actual_size > self.max_buffer_bytes) return error.BufferTooLarge;
        if (device.available_memory_bytes != 0 and actual_size_u64 > device.available_memory_bytes) {
            return error.DeviceMemoryExceeded;
        }

        const handle = try nextHandle(self.buffers.items.len);
        const bytes = try allocator.alloc(u8, actual_size);
        errdefer allocator.free(bytes);
        @memset(bytes, 0);

        try self.buffers.append(allocator, .{
            .device_handle = device_handle,
            .bytes = bytes,
        });
        return handle;
    }

    pub fn writeBuffer(
        self: *Runtime,
        buffer_handle: BufferHandle,
        offset: u32,
        data: []const u8,
    ) !void {
        const buffer = self.bufferBinding(buffer_handle) orelse return error.InvalidBufferHandle;
        const range = try checkedRange(buffer.bytes.len, offset, data.len);
        @memcpy(buffer.bytes[range.start..range.end], data);
    }

    pub fn readBuffer(
        self: *const Runtime,
        buffer_handle: BufferHandle,
        offset: u32,
        out: []u8,
    ) !void {
        const buffer = self.constBufferBinding(buffer_handle) orelse return error.InvalidBufferHandle;
        const range = try checkedRange(buffer.bytes.len, offset, out.len);
        @memcpy(out, buffer.bytes[range.start..range.end]);
    }

    pub fn dispatchComputeStub(
        self: *const Runtime,
        queue_handle: QueueHandle,
        buffer_handle: BufferHandle,
        workgroup_x: u32,
        workgroup_y: u32,
        workgroup_z: u32,
    ) !void {
        const queue = self.queueBinding(queue_handle) orelse return error.InvalidQueueHandle;
        const buffer = self.constBufferBinding(buffer_handle) orelse return error.InvalidBufferHandle;
        if (queue.device_handle != buffer.device_handle) return error.DeviceMismatch;
        if (workgroup_x == 0 or workgroup_y == 0 or workgroup_z == 0) return error.InvalidWorkgroupCount;
    }

    fn deviceBinding(self: *const Runtime, handle: DeviceHandle) ?DeviceBinding {
        const index = handleIndex(handle) orelse return null;
        if (index >= self.devices.items.len) return null;
        return self.devices.items[index];
    }

    fn queueBinding(self: *const Runtime, handle: QueueHandle) ?QueueBinding {
        const index = handleIndex(handle) orelse return null;
        if (index >= self.queues.items.len) return null;
        return self.queues.items[index];
    }

    fn bufferBinding(self: *Runtime, handle: BufferHandle) ?*Buffer {
        const index = handleIndex(handle) orelse return null;
        if (index >= self.buffers.items.len) return null;
        return &self.buffers.items[index];
    }

    fn constBufferBinding(self: *const Runtime, handle: BufferHandle) ?*const Buffer {
        const index = handleIndex(handle) orelse return null;
        if (index >= self.buffers.items.len) return null;
        return &self.buffers.items[index];
    }
};

pub fn deviceKindName(kind: DeviceKind) []const u8 {
    return switch (kind) {
        .unknown => "unknown",
        .cpu_fallback => "cpu_fallback",
        .integrated => "integrated",
        .discrete => "discrete",
        .accelerator => "accelerator",
    };
}

fn nextHandle(index: usize) !u32 {
    if (index >= std.math.maxInt(u32)) return error.ResourceLimit;
    return @intCast(index + 1);
}

fn handleIndex(handle: u32) ?usize {
    if (handle == 0) return null;
    return std.math.cast(usize, handle - 1) orelse null;
}

const Range = struct {
    start: usize,
    end: usize,
};

fn checkedRange(buffer_len: usize, offset: u32, len: usize) !Range {
    const start = std.math.cast(usize, offset) orelse return error.InvalidRange;
    const end = std.math.add(usize, start, len) catch return error.InvalidRange;
    if (end > buffer_len) return error.InvalidRange;
    return .{
        .start = start,
        .end = end,
    };
}

test "gpu capabilities hide devices when disabled" {
    const devices = [_]Device{.{ .kind = .discrete }};
    const disabled: Capabilities = .{ .enabled = false, .devices = &devices };
    const enabled: Capabilities = .{ .enabled = true, .devices = &devices };

    try std.testing.expectEqual(@as(usize, 0), disabled.deviceCount());
    try std.testing.expectEqual(@as(usize, 1), enabled.deviceCount());
    try std.testing.expectEqual(DeviceKind.discrete, enabled.device(0).?.kind);
}

test "gpu runtime opens devices queues and roundtrips buffer bytes" {
    const allocator = std.testing.allocator;
    var runtime = Runtime{};
    defer runtime.deinit(allocator);

    const caps = mockEdgeCapabilities();
    const device = try runtime.openDevice(allocator, caps, 0);
    const queue = try runtime.defaultQueue(allocator, caps, device);
    const buffer = try runtime.createBuffer(allocator, caps, device, 4);

    try runtime.writeBuffer(buffer, 0, &.{ 1, 2, 3, 4 });

    var out = [_]u8{0} ** 4;
    try runtime.readBuffer(buffer, 0, &out);
    try std.testing.expectEqualSlices(u8, &.{ 1, 2, 3, 4 }, &out);

    try runtime.dispatchComputeStub(queue, buffer, 1, 1, 1);
    try std.testing.expectError(error.InvalidWorkgroupCount, runtime.dispatchComputeStub(queue, buffer, 0, 1, 1));
}
