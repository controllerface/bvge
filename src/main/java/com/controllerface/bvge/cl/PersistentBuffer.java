package com.controllerface.bvge.cl;

public class PersistentBuffer extends ResizableBuffer
{
    public PersistentBuffer(int item_size)
    {
        super(item_size);
    }

    @Override
    long resize(long previous_capacity)
    {
        long new_pointer = GPGPU.cl_new_buffer(this.byte_capacity);
        GPGPU.cl_transfer_buffer(this.pointer, new_pointer, previous_capacity);
        release();
        return new_pointer;
    }
}
