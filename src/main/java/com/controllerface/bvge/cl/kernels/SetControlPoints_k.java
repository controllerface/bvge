package com.controllerface.bvge.cl.kernels;

public class SetControlPoints_k extends GPUKernel
{
    public enum Args
    {
        flags,
        indices,
        linear_mag,
        jump_mag,
        target,
        new_flags,
        new_index,
        new_linear_mag,
        new_jump_mag,
    }

    public SetControlPoints_k(long command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr, kernel_ptr);
    }
}
