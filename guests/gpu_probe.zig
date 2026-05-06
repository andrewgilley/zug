extern "zug_gpu" fn device_count() u32;
extern "zug_gpu" fn device_kind(index: u32) u32;
extern "zug_gpu" fn device_memory(index: u32, total_ptr: *u64, available_ptr: *u64) u32;
extern "zug_gpu" fn device_queue_count(index: u32) u32;
extern "zug_gpu" fn select_device(index: u32) u32;
extern "zug_gpu" fn selected_device() u32;
extern "zug_gpu" fn open_device(index: u32, out_device_handle: *u32) u32;
extern "zug_gpu" fn default_queue(device_handle: u32, out_queue_handle: *u32) u32;
extern "zug_gpu" fn create_buffer(device_handle: u32, size: u32, out_buffer_handle: *u32) u32;
extern "zug_gpu" fn write_buffer(buffer_handle: u32, offset: u32, src_ptr: *const u8, len: u32) u32;
extern "zug_gpu" fn read_buffer(buffer_handle: u32, offset: u32, dst_ptr: *u8, len: u32) u32;
extern "zug_gpu" fn dispatch_compute_stub(queue_handle: u32, buffer_handle: u32, workgroup_x: u32, workgroup_y: u32, workgroup_z: u32) u32;

var total_memory: u64 = 0;
var available_memory: u64 = 0;
var device_handle: u32 = 0;
var queue_handle: u32 = 0;
var buffer_handle: u32 = 0;
var upload = [_]u8{ 7, 8, 9, 10 };
var download = [_]u8{0} ** 4;

pub export fn run() u32 {
    if (device_count() == 0) return 0;
    if (device_memory(0, &total_memory, &available_memory) != 0) return 100;
    if (select_device(0) != 0) return 101;
    if (open_device(0, &device_handle) != 0) return 102;
    if (default_queue(device_handle, &queue_handle) != 0) return 103;
    if (create_buffer(device_handle, 4, &buffer_handle) != 0) return 104;
    if (write_buffer(buffer_handle, 0, &upload[0], 4) != 0) return 105;
    if (dispatch_compute_stub(queue_handle, buffer_handle, 1, 1, 1) != 0) return 106;
    if (read_buffer(buffer_handle, 0, &download[0], 4) != 0) return 107;
    if (download[0] != upload[0] or download[3] != upload[3]) return 108;

    return device_kind(0) + device_queue_count(0) + selected_device();
}
