package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPUKernel;

public class HandleMovement_k extends GPUKernel
{
    public enum Args
    {
        armature_accel,
        armature_flags,
        flags,
        indices,
        tick_budgets,
        linear_mag,
        jump_mag,
    }

    public HandleMovement_k(long command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr, kernel_ptr);
    }
}
