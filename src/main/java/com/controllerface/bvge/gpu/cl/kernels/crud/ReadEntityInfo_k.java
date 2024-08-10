package com.controllerface.bvge.gpu.cl.kernels.crud;

import com.controllerface.bvge.gpu.cl.buffers.CL_Buffer;
import com.controllerface.bvge.gpu.cl.contexts.CL_CommandQueue;
import com.controllerface.bvge.gpu.cl.kernels.GPUKernel;
import com.controllerface.bvge.gpu.cl.kernels.KernelType;
import com.controllerface.bvge.gpu.cl.programs.GPUProgram;
import com.controllerface.bvge.memory.groups.CoreBufferGroup;

import static com.controllerface.bvge.memory.types.CoreBufferType.*;

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

    public GPUKernel init(CoreBufferGroup core_buffers, CL_Buffer info_buffer)
    {
        return this.buf_arg(Args.entities, core_buffers.buffer(ENTITY))
            .buf_arg(Args.entity_accel, core_buffers.buffer(ENTITY_ACCEL))
            .buf_arg(Args.entity_motion_states, core_buffers.buffer(ENTITY_MOTION_STATE))
            .buf_arg(Args.entity_flags, core_buffers.buffer(ENTITY_FLAG))
            .buf_arg(Args.entity_animation_layers, core_buffers.buffer(ENTITY_ANIM_LAYER))
            .buf_arg(Args.entity_previous_layers, core_buffers.buffer(ENTITY_PREV_LAYER))
            .buf_arg(Args.entity_animation_time, core_buffers.buffer(ENTITY_ANIM_TIME))
            .buf_arg(Args.entity_previous_time, core_buffers.buffer(ENTITY_PREV_TIME))
            .buf_arg(Args.entity_animation_blend, core_buffers.buffer(ENTITY_ANIM_BLEND))
            .buf_arg(Args.output, info_buffer);
    }
}
