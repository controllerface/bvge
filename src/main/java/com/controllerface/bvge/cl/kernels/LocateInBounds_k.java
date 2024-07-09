package com.controllerface.bvge.cl.kernels;

public class LocateInBounds_k extends GPUKernel
{
    public enum Args
    {
        bounds_bank_data,
        in_bounds,
        counter,
        max_bound,
    }

    public LocateInBounds_k(long command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr, kernel_ptr);
    }
}
