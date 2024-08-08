package com.controllerface.bvge.gpu.cl.kernels.crud;

import com.controllerface.bvge.gpu.cl.contexts.CL_CommandQueue;
import com.controllerface.bvge.gpu.cl.kernels.GPUKernel;

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

    public UpdateSelectBlock_k(CL_CommandQueue command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr, kernel_ptr);
    }
}
