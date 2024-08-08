package com.controllerface.bvge.gpu.cl.kernels.crud;

import com.controllerface.bvge.gpu.cl.kernels.GPUKernel;

public class WriteEntityInfo_k extends GPUKernel
{
    public enum Args
    {
        entity_accel,
        entity_animation_time,
        entity_previous_time,
        entity_animation_blend,
        entity_motion_states,
        entity_animation_layers,
        entity_previous_layers,
        entity_flags,
        target,
        new_accel,
        new_anim_time,
        new_prev_time,
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
