package com.controllerface.bvge.gpu.cl.kernels.rendering;

import com.controllerface.bvge.gpu.cl.contexts.CL_CommandQueue;
import com.controllerface.bvge.gpu.cl.kernels.GPUKernel;

public class PrepareEntities_k extends GPUKernel
{
    public enum Args
    {
        points,
        vertex_vbo,
        offset,
        max_entity,
    }

    public PrepareEntities_k(CL_CommandQueue command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr, kernel_ptr);
    }
}
