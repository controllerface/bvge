package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPUKernel;

public class LocateOutOfBounds_k extends GPUKernel
{
    public enum Args
    {
        hull_tables,
        hull_flags,
        armature_flags,
        counter;
    }

    public LocateOutOfBounds_k(long command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr, kernel_ptr);
    }
}
