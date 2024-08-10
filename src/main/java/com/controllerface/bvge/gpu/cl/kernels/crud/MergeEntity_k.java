package com.controllerface.bvge.gpu.cl.kernels.crud;

import com.controllerface.bvge.gpu.cl.contexts.CL_CommandQueue;
import com.controllerface.bvge.gpu.cl.kernels.GPUKernel;
import com.controllerface.bvge.gpu.cl.kernels.KernelType;
import com.controllerface.bvge.gpu.cl.programs.GPUProgram;
import com.controllerface.bvge.memory.GPUCoreMemory;
import com.controllerface.bvge.memory.groups.CoreBufferGroup;

import static com.controllerface.bvge.memory.types.CoreBufferType.*;

public class MergeEntity_k extends GPUKernel
{
    public enum Args
    {
        entities_in,
        entity_animation_time_in,
        entity_previous_time_in,
        entity_motion_states_in,
        entity_animation_layers_in,
        entity_previous_layers_in,
        entity_hull_tables_in,
        entity_bone_tables_in,
        entity_masses_in,
        entity_root_hulls_in,
        entity_model_indices_in,
        entity_model_transforms_in,
        entity_types_in,
        entity_flags_in,
        entities_out,
        entity_animation_time_out,
        entity_previous_time_out,
        entity_motion_states_out,
        entity_animation_layers_out,
        entity_previous_layers_out,
        entity_hull_tables_out,
        entity_bone_tables_out,
        entity_masses_out,
        entity_root_hulls_out,
        entity_model_indices_out,
        entity_model_transforms_out,
        entity_types_out,
        entity_flags_out,
        entity_offset,
        hull_offset,
        armature_bone_offset,
        max_entity,
    }

    public MergeEntity_k(CL_CommandQueue command_queue_ptr, GPUProgram program)
    {
        super(command_queue_ptr, program.get_kernel(KernelType.merge_entity));
    }

    public GPUKernel init(GPUCoreMemory core_memory, CoreBufferGroup core_buffers)
    {
        return this.buf_arg(Args.entities_in, core_buffers.buffer(ENTITY))
            .buf_arg(Args.entity_animation_time_in, core_buffers.buffer(ENTITY_ANIM_TIME))
            .buf_arg(Args.entity_previous_time_in, core_buffers.buffer(ENTITY_PREV_TIME))
            .buf_arg(Args.entity_motion_states_in, core_buffers.buffer(ENTITY_MOTION_STATE))
            .buf_arg(Args.entity_animation_layers_in, core_buffers.buffer(ENTITY_ANIM_LAYER))
            .buf_arg(Args.entity_previous_layers_in, core_buffers.buffer(ENTITY_PREV_LAYER))
            .buf_arg(Args.entity_hull_tables_in, core_buffers.buffer(ENTITY_HULL_TABLE))
            .buf_arg(Args.entity_bone_tables_in, core_buffers.buffer(ENTITY_BONE_TABLE))
            .buf_arg(Args.entity_masses_in, core_buffers.buffer(ENTITY_MASS))
            .buf_arg(Args.entity_root_hulls_in, core_buffers.buffer(ENTITY_ROOT_HULL))
            .buf_arg(Args.entity_model_indices_in, core_buffers.buffer(ENTITY_MODEL_ID))
            .buf_arg(Args.entity_model_transforms_in, core_buffers.buffer(ENTITY_TRANSFORM_ID))
            .buf_arg(Args.entity_types_in, core_buffers.buffer(ENTITY_TYPE))
            .buf_arg(Args.entity_flags_in, core_buffers.buffer(ENTITY_FLAG))
            .buf_arg(Args.entities_out, core_memory.get_buffer(ENTITY))
            .buf_arg(Args.entity_animation_time_out, core_memory.get_buffer(ENTITY_ANIM_TIME))
            .buf_arg(Args.entity_previous_time_out, core_memory.get_buffer(ENTITY_PREV_TIME))
            .buf_arg(Args.entity_motion_states_out, core_memory.get_buffer(ENTITY_MOTION_STATE))
            .buf_arg(Args.entity_animation_layers_out, core_memory.get_buffer(ENTITY_ANIM_LAYER))
            .buf_arg(Args.entity_previous_layers_out, core_memory.get_buffer(ENTITY_PREV_LAYER))
            .buf_arg(Args.entity_hull_tables_out, core_memory.get_buffer(ENTITY_HULL_TABLE))
            .buf_arg(Args.entity_bone_tables_out, core_memory.get_buffer(ENTITY_BONE_TABLE))
            .buf_arg(Args.entity_masses_out, core_memory.get_buffer(ENTITY_MASS))
            .buf_arg(Args.entity_root_hulls_out, core_memory.get_buffer(ENTITY_ROOT_HULL))
            .buf_arg(Args.entity_model_indices_out, core_memory.get_buffer(ENTITY_MODEL_ID))
            .buf_arg(Args.entity_model_transforms_out, core_memory.get_buffer(ENTITY_TRANSFORM_ID))
            .buf_arg(Args.entity_types_out, core_memory.get_buffer(ENTITY_TYPE))
            .buf_arg(Args.entity_flags_out, core_memory.get_buffer(ENTITY_FLAG));
    }
}
