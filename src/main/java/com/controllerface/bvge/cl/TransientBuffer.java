package com.controllerface.bvge.cl;

public class TransientBuffer extends ResizableBuffer
{
    public TransientBuffer(int item_size)
    {
        super(item_size);
    }

    public void ensure_capacity(long total_item_capacity)
    {
        var required_capacity = item_size * total_item_capacity;
        if (required_capacity <= this.byte_capacity) return;

        this.byte_capacity = required_capacity;
        release();
        this.pointer = GPGPU.cl_new_buffer(this.byte_capacity);
        update_registered_kernels();
    }
}
