package com.controllerface.bvge.cl.kernels;

public class WriteEntityInfo_k extends GPUKernel
{
    public enum Args
    {
        entity_accel,
        entity_animation_elapsed,
        entity_animation_blend,
        entity_motion_states,
        entity_animation_indices,
        entity_flags,
        target,
        new_accel,
        new_anim_elapsed,
        new_anim_blend,
        new_motion_state,
        new_anim_indices,
        new_flags,
    }

    public WriteEntityInfo_k(long command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr, kernel_ptr);
    }
}
