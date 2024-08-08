package com.controllerface.bvge.gpu.cl.buffers;

import com.controllerface.bvge.gpu.cl.GPGPU;

public class TransientBuffer extends ResizableBuffer
{
    public TransientBuffer(long queue_ptr, int item_size, long initial_capacity)
    {
        super(queue_ptr, item_size, initial_capacity);
    }

    public void ensure_capacity(long total_item_capacity)
    {
        var required_capacity = item_size * total_item_capacity;
        if (required_capacity <= this.byte_capacity) return;

        this.byte_capacity = required_capacity * 2;
        release();
        this.buffer_pointer = GPGPU.cl_new_buffer(this.byte_capacity);
        update_registered_kernels();
    }
}
