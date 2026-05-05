(module
  (import "wasi_nn" "load_graph"
    (func $load_graph
      (param i32 i32 i32 i32 i32)
      (result i32)))
  (import "wasi_nn" "init_execution_context"
    (func $init_execution_context
      (param i32 i32)
      (result i32)))
  (import "wasi_nn" "set_input_by_index"
    (func $set_input_by_index
      (param i32 i32 i32 i32 i32 i32 i32)
      (result i32)))
  (import "wasi_nn" "compute"
    (func $compute
      (param i32)
      (result i32)))

  (memory (export "memory") 32)

  (data (i32.const 200000)
    "\01\00\00\00\00\00\00\00"
    "\01\00\00\00\00\00\00\00"
    "\1c\00\00\00\00\00\00\00"
    "\1c\00\00\00\00\00\00\00")

  (func (export "run") (param $model_len i32) (result i32)
    i32.const 1024
    local.get $model_len
    i32.const 0
    i32.const 0
    i32.const 16
    call $load_graph
    drop

    i32.const 16
    i32.load
    i32.const 20
    call $init_execution_context
    drop

    i32.const 20
    i32.load
    i32.const 0
    i32.const 1
    i32.const 200000
    i32.const 4
    i32.const 201000
    i32.const 3136
    call $set_input_by_index
    drop

    i32.const 20
    i32.load
    call $compute))
