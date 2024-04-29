package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPUKernel;

public class SatSortCount_k extends GPUKernel
{
    public enum Args
    {
        candidates,
        hull_flags,
        counter,
    }

    public SatSortCount_k(long command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr, kernel_ptr);
    }
}
