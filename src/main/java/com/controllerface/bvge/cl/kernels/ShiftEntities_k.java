package com.controllerface.bvge.cl.kernels;

public class ShiftEntities_k extends GPUKernel
{
    public enum Args
    {
        entities,
        x_shift,
        y_shift,
        max_entity,
    }

    public ShiftEntities_k(long command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr, kernel_ptr);
    }
}
