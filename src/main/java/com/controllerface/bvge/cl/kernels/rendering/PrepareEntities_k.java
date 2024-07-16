package com.controllerface.bvge.cl.kernels.rendering;

import com.controllerface.bvge.cl.kernels.GPUKernel;

public class PrepareEntities_k extends GPUKernel
{
    public enum Args
    {
        points,
        vertex_vbo,
        offset,
        max_entity,
    }

    public PrepareEntities_k(long command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr, kernel_ptr);
    }
}
