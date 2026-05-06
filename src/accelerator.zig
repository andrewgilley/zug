const std = @import("std");

pub const BackendKind = enum(u32) {
    cpu = 0,
    cuda = 1,
    tensorrt = 2,
    rocm = 3,
    vulkan = 4,
    metal = 5,
    directml = 6,
    openvino = 7,
    coreml = 8,
    nnapi = 9,
    webgpu = 10,
    edge_tpu = 11,
};

pub const BackendStatus = enum {
    unavailable,
    planned,
    enabled,
};

pub const Feature = enum {
    graph_execution,
    tensor_buffers,
    fp32,
    fp16,
    int8,
    conv,
    matmul,
    zero_copy,
};

pub const Backend = struct {
    kind: BackendKind,
    status: BackendStatus = .unavailable,
    device_count: usize = 0,
    total_memory_bytes: u64 = 0,
    available_memory_bytes: u64 = 0,
    features: []const Feature = &.{},

    pub fn enabled(self: Backend) bool {
        return self.status == .enabled;
    }

    pub fn supportsFeature(self: Backend, feature: Feature) bool {
        for (self.features) |supported| {
            if (supported == feature) return true;
        }

        return false;
    }

    pub fn supportsGraphExecution(self: Backend) bool {
        return self.enabled() and self.supportsFeature(.graph_execution);
    }
};

pub const Capabilities = struct {
    backends: []const Backend = &default_backends,

    pub fn backend(self: Capabilities, kind: BackendKind) ?Backend {
        for (self.backends) |item| {
            if (item.kind == kind) return item;
        }

        return null;
    }

    pub fn supportsGraphExecution(self: Capabilities, kind: BackendKind) bool {
        const item = self.backend(kind) orelse return false;
        return item.supportsGraphExecution();
    }

    pub fn defaultGpuBackend(self: Capabilities) ?BackendKind {
        const preferred = [_]BackendKind{
            .cuda,
            .tensorrt,
            .rocm,
            .vulkan,
            .directml,
            .metal,
            .coreml,
            .nnapi,
            .webgpu,
            .openvino,
            .edge_tpu,
        };

        for (preferred) |kind| {
            if (self.supportsGraphExecution(kind)) return kind;
        }

        return null;
    }
};

const cpu_features = [_]Feature{
    .graph_execution,
    .tensor_buffers,
    .fp32,
    .conv,
    .matmul,
};

const planned_gpu_features = [_]Feature{
    .tensor_buffers,
    .fp32,
    .fp16,
    .int8,
    .conv,
    .matmul,
    .zero_copy,
};

pub const default_backends = [_]Backend{
    .{
        .kind = .cpu,
        .status = .enabled,
        .device_count = 1,
        .features = &cpu_features,
    },
    .{ .kind = .cuda, .status = .planned, .features = &planned_gpu_features },
    .{ .kind = .tensorrt, .status = .planned, .features = &planned_gpu_features },
    .{ .kind = .rocm, .status = .planned, .features = &planned_gpu_features },
    .{ .kind = .vulkan, .status = .planned, .features = &planned_gpu_features },
    .{ .kind = .metal, .status = .planned, .features = &planned_gpu_features },
    .{ .kind = .directml, .status = .planned, .features = &planned_gpu_features },
    .{ .kind = .openvino, .status = .planned, .features = &planned_gpu_features },
    .{ .kind = .coreml, .status = .planned, .features = &planned_gpu_features },
    .{ .kind = .nnapi, .status = .planned, .features = &planned_gpu_features },
    .{ .kind = .webgpu, .status = .planned, .features = &planned_gpu_features },
    .{ .kind = .edge_tpu, .status = .planned, .features = &planned_gpu_features },
};

const mock_edge_features = [_]Feature{
    .graph_execution,
    .tensor_buffers,
    .fp32,
    .fp16,
    .conv,
    .matmul,
};

pub const mock_edge_backends = [_]Backend{
    .{
        .kind = .cpu,
        .status = .enabled,
        .device_count = 1,
        .features = &cpu_features,
    },
    .{
        .kind = .vulkan,
        .status = .enabled,
        .device_count = 1,
        .total_memory_bytes = 512 * 1024 * 1024,
        .available_memory_bytes = 384 * 1024 * 1024,
        .features = &mock_edge_features,
    },
    .{
        .kind = .directml,
        .status = .enabled,
        .device_count = 1,
        .total_memory_bytes = 512 * 1024 * 1024,
        .available_memory_bytes = 384 * 1024 * 1024,
        .features = &mock_edge_features,
    },
};

pub fn defaultCapabilities() Capabilities {
    return .{ .backends = &default_backends };
}

pub fn mockEdgeCapabilities() Capabilities {
    return .{ .backends = &mock_edge_backends };
}

pub fn backendKindName(kind: BackendKind) []const u8 {
    return switch (kind) {
        .cpu => "cpu",
        .cuda => "cuda",
        .tensorrt => "tensorrt",
        .rocm => "rocm",
        .vulkan => "vulkan",
        .metal => "metal",
        .directml => "directml",
        .openvino => "openvino",
        .coreml => "coreml",
        .nnapi => "nnapi",
        .webgpu => "webgpu",
        .edge_tpu => "edge_tpu",
    };
}

pub fn backendStatusName(status: BackendStatus) []const u8 {
    return switch (status) {
        .unavailable => "unavailable",
        .planned => "planned",
        .enabled => "enabled",
    };
}

pub fn featureName(feature: Feature) []const u8 {
    return switch (feature) {
        .graph_execution => "graph_execution",
        .tensor_buffers => "tensor_buffers",
        .fp32 => "fp32",
        .fp16 => "fp16",
        .int8 => "int8",
        .conv => "conv",
        .matmul => "matmul",
        .zero_copy => "zero_copy",
    };
}

test "default accelerator capabilities expose cpu execution and planned popular backends" {
    const caps = defaultCapabilities();

    try std.testing.expect(caps.supportsGraphExecution(.cpu));
    try std.testing.expect(!caps.supportsGraphExecution(.cuda));
    try std.testing.expectEqual(BackendStatus.planned, caps.backend(.vulkan).?.status);
    try std.testing.expectEqualStrings("directml", backendKindName(.directml));
}

test "mock edge accelerator capabilities choose a gpu backend" {
    const caps = mockEdgeCapabilities();

    try std.testing.expect(caps.supportsGraphExecution(.vulkan));
    try std.testing.expectEqual(BackendKind.vulkan, caps.defaultGpuBackend().?);
}
