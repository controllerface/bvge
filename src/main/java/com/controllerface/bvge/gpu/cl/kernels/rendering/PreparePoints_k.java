package com.controllerface.bvge.gpu.cl.kernels.rendering;

import com.controllerface.bvge.gpu.cl.kernels.GPUKernel;

public class PreparePoints_k extends GPUKernel
{
    public enum Args
    {
        points,
        anti_gravity,
        vertex_vbo,
        color_vbo,
        offset,
        max_point,
    }

    public PreparePoints_k(long command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr, kernel_ptr);
    }
}
