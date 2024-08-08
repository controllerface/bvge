package com.controllerface.bvge.gpu.cl.buffers;

import com.controllerface.bvge.gpu.GPU;
import com.controllerface.bvge.gpu.cl.GPGPU;
import com.controllerface.bvge.gpu.cl.contexts.CL_CommandQueue;

import static com.controllerface.bvge.gpu.cl.buffers.CL_DataTypes.cl_float16;

public class PersistentBuffer extends ResizableBuffer
{
    public PersistentBuffer(CL_CommandQueue cmd_queue, int item_size, long initial_capacity)
    {
        super(cmd_queue, item_size, initial_capacity);
        clear();
    }

    public PersistentBuffer(CL_CommandQueue cmd_queue, int item_size)
    {
        this(cmd_queue, item_size, DEFAULT_ITEM_CAPACITY);
    }

    public void ensure_capacity(long total_item_capacity)
    {
        var required_capacity = item_size * total_item_capacity;
        if (required_capacity <= this.byte_capacity) return;

        long previous_capacity = this.byte_capacity;
        while (this.byte_capacity < required_capacity)
        {
            // todo: define different expansion strategies for different buffer types
            if (this.item_size == cl_float16.size()) this.byte_capacity += (long)this.item_size * 8192L;
            else this.byte_capacity += (long)this.item_size * 32768L;
        }

        var new_buffer = GPU.CL.new_buffer(GPGPU.compute.context, this.byte_capacity);
        GPGPU.cl_zero_buffer(cmd_queue.ptr(), new_buffer.ptr(), this.byte_capacity);
        GPGPU.cl_transfer_buffer(cmd_queue.ptr(), this.buffer.ptr(), new_buffer.ptr(), previous_capacity);

        release();
        this.buffer = new_buffer;
        update_registered_kernels();
    }

    @Override
    public void copy_from(ResizableBuffer source)
    {
        release();
        this.byte_capacity = source.byte_capacity;
        var new_buffer = GPU.CL.new_buffer(GPGPU.compute.context, this.byte_capacity);
        GPGPU.cl_transfer_buffer(cmd_queue.ptr(), source.buffer.ptr(), new_buffer.ptr(), source.byte_capacity);
        this.buffer = new_buffer;
        update_registered_kernels();
    }
}
