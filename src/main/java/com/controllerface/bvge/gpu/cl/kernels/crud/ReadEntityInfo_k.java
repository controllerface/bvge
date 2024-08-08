package com.controllerface.bvge.gpu.cl.kernels.crud;

import com.controllerface.bvge.gpu.cl.contexts.CL_CommandQueue;
import com.controllerface.bvge.gpu.cl.kernels.GPUKernel;

public class ReadEntityInfo_k extends GPUKernel
{
    public enum Args
    {
        entities,
        entity_accel,
        entity_motion_states,
        entity_flags,
        entity_animation_layers,
        entity_previous_layers,
        entity_animation_time,
        entity_previous_time,
        entity_animation_blend,
        output,
        target,
    }

    public ReadEntityInfo_k(CL_CommandQueue command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr, kernel_ptr);
    }
}
