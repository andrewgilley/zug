(module
  (import "wasi_nn" "compute" (func $compute (param i32) (result i32)))

  (func (export "run") (result i32)
    i32.const 999
    call $compute))
