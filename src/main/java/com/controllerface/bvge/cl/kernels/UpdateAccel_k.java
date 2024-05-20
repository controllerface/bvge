package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPUKernel;

public class UpdateAccel_k extends GPUKernel
{
    public enum Args
    {
        entity_accel,
        target,
        new_value;
    }

    public UpdateAccel_k(long command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr, kernel_ptr);
    }
}
