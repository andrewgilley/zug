# zug

`zug` is a small WebAssembly runtime written in Zig. It parses, validates,
instantiates, and interprets core WebAssembly modules without delegating
execution to another runtime.

The current runtime includes:

- Core WebAssembly module parsing and validation.
- An interpreter covering integer, floating-point, memory, table, control-flow,
  bulk-memory, and selected SIMD instructions.
- WASI Preview 1 host calls for arguments, environment, clocks, randomness,
  standard I/O, process exit, and read-only preopened files.
- Component binary parsing and a small WIT descriptor/generator.
- Optional runtime manifests for export, import, and memory requirements.

## Build and test

```powershell
zig build
zig build test
zig build guests
zig build test-guests
```

## CLI

Run an exported function:

```powershell
zig build run -- run zig-out/basic.wasm --arg 7
```

Check whether a module uses features supported by the runtime:

```powershell
zig build run -- check module.wasm --export run
```

Generate the included component smoke-test interface:

```powershell
zig build gen-wit
```

## Runtime manifest

A line-oriented manifest can constrain a module run:

```text
name = example
runtime = zug-0.1
export = run
min_memory_bytes = 64KiB
max_memory_bytes = 16MiB
requires_import = wasi_snapshot_preview1.fd_write
```

Pass it with `zug run module.wasm --manifest module.zugmanifest`.
