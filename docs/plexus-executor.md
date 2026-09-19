# Local Plexus executor bridge

`zug-plexus` implements protocols `plexus-executor/1` and, for linked modules,
`plexus-executor/2`. Plexus owns plans, archived
artifacts, expected answers and interpretation. The worker only executes a pinned
core Wasm module and reports observations. It never accepts expected answers.

Build and check with Zig 0.16:

```sh
zig build plexus test-plexus
python3 tests/plexus_bridge.py
zig-out/bin/zug-plexus describe
```

The executable is `zig-out/bin/zug-plexus`. A plain `zig build` installs it
alongside the `zug` CLI, and `zig build test` runs the runtime, CLI and bridge
tests together.

## Process contract

Invoke without a shell:

```text
zug-plexus describe
zug-plexus execute /absolute/request.json
```

Each process writes one JSON object and a newline to stdout. Malformed requests,
incorrect digests, file errors and protocol violations exit nonzero, with a
short diagnostic on stderr. Valid unsupported workloads return an `unsupported`
execution with exit status zero. There is no long-running server or networking.

Description response:

```json
{"schema_version":1,"protocol":"plexus-executor/1","backend":"zug","version":"0.0.1","capabilities":{"fuel":true,"memory_limit":true,"max_results":1,"memory_input":true,"memory_bytes":true,"linked_modules":true}}
```

`linked_modules` is an additive capability: the worker also accepts
`plexus-executor/2` execute requests, described below.

Execution request:

```json
{
  "schema_version": 1,
  "protocol": "plexus-executor/1",
  "module_path": "/absolute/module.wasm",
  "module_digest": "sha256:<64 lowercase hex digits>",
  "request": {
    "export": "run",
    "result_count": 1,
    "cases": [{"name":"example","arguments":[],"memory":null}],
    "limits": {"fuel_per_case":10000,"memory_bytes":65536}
  }
}
```

The worker reads the bytes once, verifies the SHA-256 digest and executes those
same bytes. Paths must be absolute. JSON fields are strict. Input JSON is bounded
at 8 MiB and the module at 4 MiB. Cases must have unique nonempty names; there
must be 1–64 cases, 0–16 i32 arguments and 0–1 i32 results. Fuel must be
1–10,000,000 and the memory cap 64 KiB–64 MiB.

Each optional memory input has shape
`{"export":"memory","offset":0,"utf8":"one\ntwo\n"}` or
`{"export":"memory","offset":0,"bytes":[0,255,128,1]}`. Exactly one payload
is required; raw bytes are integers 0..255 and preserve non-text data.
The additive `memory_bytes` capability advertises the binary form; older workers
without that flag still support their original UTF-8 inputs. Payloads are written
after the start function and before the export invocation. Writes target the
named exported memory and must fit its current size. Payload size is limited to
1 MiB. An input setup failure is reported at stage `input` and is not a guest trap.

A fresh instance is created for every case. Memory starts at its declared
minimum (including zero pages), not at the host cap. `memory.grow` respects both
the module maximum and the host cap, returning -1 when growth is denied. Tables,
host imports, multiple memories, shared memory, memory64 and custom page sizes
are outside this bridge's current profile. Component execution is unsupported.
The existing Zug validator also rejects instructions it does not implement.

Execution response:

```json
{
  "schema_version": 1,
  "protocol": "plexus-executor/1",
  "module_digest": "sha256:<verified digest>",
  "execution": {
    "kind": "observed",
    "cases": [{"name":"example","outcome":{"kind":"returned","values":[7]},"fuel_consumed":2}]
  }
}
```

Possible outcomes are:

- `returned`: `values` is an array of signed i32 values.
- `fuel_exhausted`: `stage` is `initialization` or `invocation`.
- `trapped`: `stage`, canonical `code`, and runtime `diagnostic`.
- `failed`: `stage` and `diagnostic`, representing an operational failure.

Stages are `initialization`, `input`, and `invocation`. Known traps use codes
matching the reference adapter: `UnreachableCodeReached`, `MemoryOutOfBounds`,
`IntegerDivisionByZero`, `IntegerOverflow`, `BadConversionToInteger`, and
`StackOverflow`. Other failures remain operational failures; they are not
silently promoted to guest traps.

Unsupported response replaces `execution` with
`{"kind":"unsupported","reason":"..."}`. It is not a contradiction of the
Plexus hypothesis. The caller should retain the request, module, runtime binary
identity and response with its experiment provenance.

## Resource and reproducibility boundaries

Fuel counts interpreted instructions and includes the start function plus the
exported call. It does not measure elapsed time, parser/validator work, native
bulk-memory work, or match Wasmtime's fuel accounting. Fuel consumption is useful
provenance within one pinned backend; compare observable outcomes across
backends, not instruction counts. The bridge's one-instruction accounting and
Zug's call-stack limit can legitimately produce different resource outcomes.

The memory cap applies to guest linear memory, not all process allocations.
Runtime scratch state is discarded between cases. This bridge is a local
experimental worker, not a hardened untrusted-code sandbox. The caller should
apply a process wall-clock deadline and bound captured output. Fuel alone does
not bound parsing or validation of malformed modules.

## Linked modules (`plexus-executor/2`)

An execute request whose `protocol` is `plexus-executor/2` names up to eight
modules instead of one:

```json
{
  "schema_version": 1,
  "protocol": "plexus-executor/2",
  "modules": [
    {"name": "provider", "module_path": "/abs/provider.wasm", "module_digest": "sha256:..."},
    {"name": "consumer", "module_path": "/abs/consumer.wasm", "module_digest": "sha256:..."}
  ],
  "request": {
    "entry": {"module": "consumer", "export": "run"},
    "result_count": 1,
    "cases": [{"name": "all-high", "arguments": [0],
               "memory": {"module": "consumer", "export": "memory", "offset": 0, "bytes": [255, 255, 255, 255]}}],
    "limits": {"fuel_per_case": 10000, "memory_bytes": 65536}
  }
}
```

For every case the worker instantiates the modules in order. It binds each
module's function imports to exports of an **earlier** module whose name
equals the import module, and checks that the parameter and result types are
equal. Nothing is granted from the host. A linked call runs in the callee's
own instance and memory, and draws on the same fuel budget. That budget covers
every module's start function and the entry call. `memory_bytes` bounds each
memory.

The response echoes `modules` (names and digests, in order) in place of
`module_digest`. Outcomes gain the stage `link`: `UnresolvedFunctionImport`,
`IncompatibleImportType` or `UnsupportedLinkedImportKind` means the modules
could not be linked as declared. Imported memories, tables and globals are
reported as `unsupported`. The schemas and fixtures live in the workspace's
`contracts/plexus-executor-2`.
