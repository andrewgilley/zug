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

const message = "guest:wasi\n";

pub export fn run() u32 {
    var iov = IOVec{
        .buf = message.ptr,
        .buf_len = message.len,
    };
    var written: usize = 0;

    return fd_write(1, &iov, 1, &written);
}
