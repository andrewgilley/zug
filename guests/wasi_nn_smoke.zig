extern "wasi_nn" fn compute(context: u32) u32;

pub export fn run() u32 {
    return compute(999);
}
