package com.controllerface.bvge.cl.kernels;

public class ShiftPoints_k extends GPUKernel
{
    public enum Args
    {
        points,
        x_shift,
        y_shift,
        max_point,
    }

    public ShiftPoints_k(long command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr, kernel_ptr);
    }
}
