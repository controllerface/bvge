package com.controllerface.bvge.cl.kernels;

public class PreparePoints_k extends GPUKernel
{
    public enum Args
    {
        points,
        anti_gravity,
        vertex_vbo,
        color_vbo,
        offset;
    }

    public PreparePoints_k(long command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr, kernel_ptr);
    }
}
