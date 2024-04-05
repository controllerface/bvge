package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPUKernel;

public class SetControlPoints_k extends GPUKernel
{
    public enum Args
    {
        flags,
        indices,
        tick_budgets,
        linear_mag,
        jump_mag,
        target,
        new_flags,
        new_index,
        new_tick_budget,
        new_linear_mag,
        new_jump_mag,
    }

    public SetControlPoints_k(long command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr, kernel_ptr);
    }
}
