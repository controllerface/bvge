package com.controllerface.bvge.cl.kernels.crud;

import com.controllerface.bvge.cl.kernels.GPUKernel;

public class UpdateSelectBlock_k extends GPUKernel
{
    public enum Args
    {
        entity_flags,
        hull_uv_offsets,
        entity_hull_tables,
        target,
        new_value;
    }

    public UpdateSelectBlock_k(long command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr, kernel_ptr);
    }
}
