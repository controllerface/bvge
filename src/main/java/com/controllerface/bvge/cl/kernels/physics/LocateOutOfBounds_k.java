package com.controllerface.bvge.cl.kernels.physics;

import com.controllerface.bvge.cl.kernels.GPUKernel;

public class LocateOutOfBounds_k extends GPUKernel
{
    public enum Args
    {
        entity_flags,
    }

    public LocateOutOfBounds_k(long command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr, kernel_ptr);
    }
}
