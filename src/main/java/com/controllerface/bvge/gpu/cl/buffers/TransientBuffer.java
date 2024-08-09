package com.controllerface.bvge.gpu.cl.buffers;

import com.controllerface.bvge.gpu.GPU;
import com.controllerface.bvge.gpu.cl.contexts.CL_CommandQueue;

public class TransientBuffer extends ResizableBuffer
{
    public TransientBuffer(CL_CommandQueue cmd_queue, int item_size, long initial_capacity)
    {
        super(cmd_queue, item_size, initial_capacity);
    }

    public void ensure_capacity(long total_item_capacity)
    {
        var required_capacity = item_size * total_item_capacity;
        if (required_capacity <= this.byte_capacity) return;

        this.byte_capacity = required_capacity * 2;
        release();
        this.buffer = GPU.CL.new_buffer(GPU.compute.context, this.byte_capacity);
        update_registered_kernels();
    }
}
