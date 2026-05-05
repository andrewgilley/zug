const graph_handle_ptr: u32 = 16;
const context_handle_ptr: u32 = 20;
const output_dtype_ptr: u32 = 28;
const output_shape_len_ptr: u32 = 32;
const output_byte_len_ptr: u32 = 36;
const output_bytes_written_ptr: u32 = 40;
const input_shape_ptr: u32 = 200_000;
const input_data_ptr: u32 = 201_000;
const output_shape_ptr: u32 = 300_000;
const output_data_ptr: u32 = 300_100;

const Status = enum(u32) {
    ok = 0,
};

const IOVec = extern struct {
    buf: [*]const u8,
    buf_len: usize,
};

extern "wasi_snapshot_preview1" fn fd_write(
    fd: u32,
    iovs: *const IOVec,
    iovs_len: usize,
    nwritten: *usize,
) u32;

extern "zug_nn" fn load_preloaded_graph(
    encoding: u32,
    target: u32,
    out_graph: u32,
) u32;

extern "wasi_nn" fn init_execution_context(
    graph: u32,
    out_context: u32,
) u32;

extern "wasi_nn" fn set_input_by_index(
    context: u32,
    input_index: u32,
    dtype: u32,
    shape: u32,
    shape_len: u32,
    data: u32,
    data_len: u32,
) u32;

extern "wasi_nn" fn compute(context: u32) u32;

extern "wasi_nn" fn get_output_descriptor(
    context: u32,
    output_index: u32,
    out_dtype: u32,
    out_shape: u32,
    shape_capacity: u32,
    out_shape_len: u32,
    out_byte_len: u32,
) u32;

extern "wasi_nn" fn get_output(
    context: u32,
    output_index: u32,
    out_data: u32,
    out_data_len: u32,
    out_bytes_written: u32,
) u32;

const done_message = "guest:inference\n";

pub export fn run() u32 {
    writeInputShape();

    var status = load_preloaded_graph(0, 0, graph_handle_ptr);
    if (status != @intFromEnum(Status.ok)) return status;

    status = init_execution_context(readU32(graph_handle_ptr), context_handle_ptr);
    if (status != @intFromEnum(Status.ok)) return status;

    status = set_input_by_index(
        readU32(context_handle_ptr),
        0,
        1,
        input_shape_ptr,
        4,
        input_data_ptr,
        28 * 28 * @sizeOf(f32),
    );
    if (status != @intFromEnum(Status.ok)) return status;

    status = compute(readU32(context_handle_ptr));
    if (status != @intFromEnum(Status.ok)) return status;

    status = get_output_descriptor(
        readU32(context_handle_ptr),
        0,
        output_dtype_ptr,
        output_shape_ptr,
        2,
        output_shape_len_ptr,
        output_byte_len_ptr,
    );
    if (status != @intFromEnum(Status.ok)) return status;

    status = get_output(
        readU32(context_handle_ptr),
        0,
        output_data_ptr,
        10 * @sizeOf(f32),
        output_bytes_written_ptr,
    );
    if (status != @intFromEnum(Status.ok)) return status;

    _ = writeStdout(done_message);
    return status;
}

fn writeInputShape() void {
    writeU64(input_shape_ptr, 1);
    writeU64(input_shape_ptr + 8, 1);
    writeU64(input_shape_ptr + 16, 28);
    writeU64(input_shape_ptr + 24, 28);
}

fn readU32(address: u32) u32 {
    const value: *const u32 = @ptrFromInt(address);
    return value.*;
}

fn writeU64(address: u32, value: u64) void {
    const target: *u64 = @ptrFromInt(address);
    target.* = value;
}

fn writeStdout(message: []const u8) u32 {
    var iov = IOVec{
        .buf = message.ptr,
        .buf_len = message.len,
    };
    var written: usize = 0;

    return fd_write(1, &iov, 1, &written);
}
