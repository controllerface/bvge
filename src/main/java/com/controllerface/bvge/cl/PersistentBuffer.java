package com.controllerface.bvge.cl;

public class PersistentBuffer extends ResizableBuffer
{
    public PersistentBuffer(int item_size, long initial_capacity)
    {
        super(item_size, initial_capacity);
        clear();
    }

    public PersistentBuffer(int item_size)
    {
        super(item_size);
        clear();
    }

    public void ensure_capacity(long total_item_capacity)
    {
        var required_capacity = item_size * total_item_capacity;
        if (required_capacity <= this.byte_capacity) return;

        long previous_capacity = this.byte_capacity;
        while (this.byte_capacity < required_capacity)
        {
            // todo: define different expansion strategies for different buffer types
            if (this.item_size == CLSize.cl_float16) this.byte_capacity += (long)this.item_size * 8192L;
            else this.byte_capacity += (long)this.item_size * 32768L;
        }

        long new_pointer = GPGPU.cl_new_buffer(this.byte_capacity);
        GPGPU.cl_zero_buffer(new_pointer, this.byte_capacity);
        GPGPU.cl_transfer_buffer(this.pointer, new_pointer, previous_capacity);

        release();
        this.pointer = new_pointer;
        update_registered_kernels();
    }
}
