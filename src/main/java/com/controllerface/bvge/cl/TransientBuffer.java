package com.controllerface.bvge.cl;

public class TransientBuffer extends ResizableBuffer
{
    public TransientBuffer(int item_size)
    {
        super(item_size);
    }

    @Override
    long resize(long previous_capacity)
    {
        release();
        return GPGPU.cl_new_buffer(this.byte_capacity);
    }
}
