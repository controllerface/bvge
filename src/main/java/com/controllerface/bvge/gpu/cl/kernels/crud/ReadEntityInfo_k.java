package com.controllerface.bvge.gpu.cl.kernels.crud;

import com.controllerface.bvge.gpu.cl.contexts.CL_CommandQueue;
import com.controllerface.bvge.gpu.cl.kernels.GPUKernel;
import com.controllerface.bvge.gpu.cl.kernels.KernelType;
import com.controllerface.bvge.gpu.cl.programs.GPUProgram;

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

    public ReadEntityInfo_k(CL_CommandQueue command_queue_ptr, GPUProgram program)
    {
        super(command_queue_ptr, program.get_kernel(KernelType.read_entity_info));
    }
}
