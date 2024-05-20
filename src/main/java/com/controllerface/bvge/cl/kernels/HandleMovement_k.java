package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPUKernel;

public class HandleMovement_k extends GPUKernel
{
    public enum Args
    {
        entities,
        entity_accel,
        entity_motion_states,
        entity_flags,
        entity_animation_indices,
        entity_animation_elapsed,
        entity_animation_blend,
        flags,
        indices,
        tick_budgets,
        linear_mag,
        jump_mag,
        dt,
    }

    public HandleMovement_k(long command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr, kernel_ptr);
    }
}
