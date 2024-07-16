package com.controllerface.bvge.cl.kernels.crud;

import com.controllerface.bvge.cl.kernels.GPUKernel;

public class WriteEntityInfo_k extends GPUKernel
{
    public enum Args
    {
        entity_accel,
        entity_animation_elapsed,
        entity_animation_blend,
        entity_motion_states,
        entity_animation_layers,
        entity_animation_previous,
        entity_flags,
        target,
        new_accel,
        new_anim_elapsed,
        new_anim_blend,
        new_motion_state,
        new_anim_layers,
        new_anim_previous,
        new_flags,
    }

    public WriteEntityInfo_k(long command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr, kernel_ptr);
    }
}
