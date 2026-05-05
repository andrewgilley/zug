(module
  (import "wasi_snapshot_preview1" "fd_write"
    (func $fd_write (param i32 i32 i32 i32) (result i32)))
  (import "wasi_nn" "load_graph"
    (func $load_graph (param i32 i32 i32 i32 i32) (result i32)))
  (import "wasi_nn" "init_execution_context"
    (func $init_execution_context (param i32 i32) (result i32)))
  (import "wasi_nn" "set_input_by_index"
    (func $set_input_by_index (param i32 i32 i32 i32 i32 i32 i32) (result i32)))
  (import "wasi_nn" "compute"
    (func $compute (param i32) (result i32)))
  (import "wasi_nn" "get_output_descriptor"
    (func $get_output_descriptor (param i32 i32 i32 i32 i32 i32 i32) (result i32)))
  (import "wasi_nn" "get_output"
    (func $get_output (param i32 i32 i32 i32 i32) (result i32)))

  (memory 1 4)

  (data (i32.const 0)
    "\40\00\00\00\0b\00\00\00"
    "\50\00\00\00\0a\00\00\00")
  (data (i32.const 64) "edge:start\0a")
  (data (i32.const 80) "edge:done\0a")
  (data (i32.const 50000)
    "\01\00\00\00\00\00\00\00"
    "\01\00\00\00\00\00\00\00"
    "\1c\00\00\00\00\00\00\00"
    "\1c\00\00\00\00\00\00\00")

  (func (export "run") (param $model_len i32) (result i32)
    (local $status i32)

    memory.size
    i32.const 1
    memory.grow
    drop

    i32.const 1
    i32.const 0
    i32.const 1
    i32.const 16
    call $fd_write
    drop

    i32.const 1024
    local.get $model_len
    i32.const 0
    i32.const 0
    i32.const 20
    call $load_graph
    drop

    i32.const 20
    i32.load
    i32.const 24
    call $init_execution_context
    drop

    i32.const 24
    i32.load
    i32.const 0
    i32.const 1
    i32.const 50000
    i32.const 4
    i32.const 50100
    i32.const 3136
    call $set_input_by_index
    drop

    i32.const 24
    i32.load
    call $compute
    drop

    i32.const 24
    i32.load
    i32.const 0
    i32.const 28
    i32.const 96
    i32.const 2
    i32.const 36
    i32.const 40
    call $get_output_descriptor
    drop

    i32.const 24
    i32.load
    i32.const 0
    i32.const 128
    i32.const 40
    i32.const 44
    call $get_output
    local.set $status

    i32.const 1
    i32.const 8
    i32.const 1
    i32.const 16
    call $fd_write
    drop

    local.get $status))
