package com.controllerface.bvge.cl.buffers;

import com.controllerface.bvge.cl.CLSize;
import com.controllerface.bvge.cl.GPGPU;

public class PersistentBuffer extends ResizableBuffer
{
    public PersistentBuffer(long queue_ptr, int item_size, long initial_capacity)
    {
        super(queue_ptr, item_size, initial_capacity);
        clear();
    }

    public PersistentBuffer(long queue_ptr, int item_size)
    {
        this(queue_ptr, item_size, DEFAULT_ITEM_CAPACITY);
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
        GPGPU.cl_zero_buffer(queue_pointer, new_pointer, this.byte_capacity);
        GPGPU.cl_transfer_buffer(queue_pointer, this.buffer_pointer, new_pointer, previous_capacity);

        release();
        this.buffer_pointer = new_pointer;
        update_registered_kernels();
    }

    @Override
    public void mirror(ResizableBuffer source)
    {
        release();
        this.byte_capacity = source.byte_capacity;
        long new_pointer = GPGPU.cl_new_buffer(this.byte_capacity);
        GPGPU.cl_transfer_buffer(queue_pointer, source.buffer_pointer, new_pointer, source.byte_capacity);
        this.buffer_pointer = new_pointer;
        update_registered_kernels();
    }
}
