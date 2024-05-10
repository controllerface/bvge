package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPUKernel;

public class HandleMovement_k extends GPUKernel
{
    public enum Args
    {
        armatures,
        armature_accel,
        armature_motion_states,
        armature_flags,
        armature_animation_indices,
        armature_animation_elapsed,
        armature_animation_blend,
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
