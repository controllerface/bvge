package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPUKernel;

public class PrepareArmatures_k extends GPUKernel
{
    public enum Args
    {
        points,
        vertex_vbo,
        offset;
    }

    public PrepareArmatures_k(long command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr, kernel_ptr);
    }
}
