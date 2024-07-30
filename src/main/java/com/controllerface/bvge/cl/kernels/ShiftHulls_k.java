package com.controllerface.bvge.cl.kernels;

public class ShiftHulls_k extends GPUKernel
{
    public enum Args
    {
        hulls,
        x_shift,
        y_shift,
        max_hull,
    }

    public ShiftHulls_k(long command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr, kernel_ptr);
    }
}
