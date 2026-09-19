//! Test root for the runtime modules; `zig build test` runs these alongside the
//! CLI and Plexus bridge tests.

test {
    _ = @import("component/wit.zig");
    _ = @import("wasm/binary.zig");
    _ = @import("wasm/compatibility.zig");
    _ = @import("wasm/component.zig");
    _ = @import("wasm/imports.zig");
    _ = @import("wasm/instance.zig");
    _ = @import("wasm/interpreter.zig");
    _ = @import("wasm/manifest.zig");
    _ = @import("wasm/module.zig");
    _ = @import("wasm/runtime.zig");
    _ = @import("wasm/store.zig");
    _ = @import("wasm/validator.zig");
}
