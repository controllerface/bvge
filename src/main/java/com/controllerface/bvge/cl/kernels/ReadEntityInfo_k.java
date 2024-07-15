package com.controllerface.bvge.cl.kernels;

public class ReadEntityInfo_k extends GPUKernel
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
        output,
        target,
    }

    public ReadEntityInfo_k(long command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr, kernel_ptr);
    }
}
