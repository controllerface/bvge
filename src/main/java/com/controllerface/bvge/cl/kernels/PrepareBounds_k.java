package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPUKernel;

public class PrepareBounds_k extends GPUKernel
{
    public enum Args
    {
        bounds,
        vbo,
        offset;
    }

    public PrepareBounds_k(long command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr, kernel_ptr);
    }
}
