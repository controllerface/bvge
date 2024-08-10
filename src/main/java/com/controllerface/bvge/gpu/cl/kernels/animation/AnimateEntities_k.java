package com.controllerface.bvge.gpu.cl.kernels.animation;

import com.controllerface.bvge.gpu.GPU;
import com.controllerface.bvge.gpu.cl.contexts.CL_CommandQueue;
import com.controllerface.bvge.gpu.cl.kernels.GPUKernel;
import com.controllerface.bvge.gpu.cl.kernels.KernelType;
import com.controllerface.bvge.gpu.cl.programs.GPUProgram;

import static com.controllerface.bvge.memory.types.CoreBufferType.*;
import static com.controllerface.bvge.memory.types.ReferenceBufferType.*;

public class AnimateEntities_k extends GPUKernel
{
    public enum Args
    {
        armature_bones,
        bone_bind_poses,
        bone_layers,
        model_transforms,
        entity_flags,
        entity_bone_reference_ids,
        entity_bone_parent_ids,
        bone_channel_tables,
        bone_pos_channel_tables,
        bone_rot_channel_tables,
        bone_scl_channel_tables,
        entity_model_transforms,
        entity_bone_tables,
        key_frames,
        frame_times,
        animation_timing_indices,
        animation_durations,
        animation_tick_rates,
        entity_animation_layers,
        entity_previous_layers,
        entity_animation_time,
        entity_previous_time,
        entity_animation_blend,
        delta_time,
        max_entity,
    }

    public AnimateEntities_k(CL_CommandQueue command_queue_ptr, GPUProgram program)
    {
        super(command_queue_ptr, program.get_kernel(KernelType.animate_entities));
    }

    public GPUKernel init()
    {
        return this.buf_arg(Args.armature_bones, GPU.memory.get_buffer(ENTITY_BONE))
            .buf_arg(Args.bone_bind_poses, GPU.memory.get_buffer(BONE_BIND_POSE))
            .buf_arg(Args.bone_layers, GPU.memory.get_buffer(BONE_LAYER))
            .buf_arg(Args.model_transforms, GPU.memory.get_buffer(MODEL_TRANSFORM))
            .buf_arg(Args.entity_flags, GPU.memory.get_buffer(ENTITY_FLAG))
            .buf_arg(Args.entity_bone_reference_ids, GPU.memory.get_buffer(ENTITY_BONE_REFERENCE_ID))
            .buf_arg(Args.entity_bone_parent_ids, GPU.memory.get_buffer(ENTITY_BONE_PARENT_ID))
            .buf_arg(Args.bone_channel_tables, GPU.memory.get_buffer(BONE_ANIM_CHANNEL_TABLE))
            .buf_arg(Args.bone_pos_channel_tables, GPU.memory.get_buffer(ANIM_POS_CHANNEL))
            .buf_arg(Args.bone_rot_channel_tables, GPU.memory.get_buffer(ANIM_ROT_CHANNEL))
            .buf_arg(Args.bone_scl_channel_tables, GPU.memory.get_buffer(ANIM_SCL_CHANNEL))
            .buf_arg(Args.entity_model_transforms, GPU.memory.get_buffer(ENTITY_TRANSFORM_ID))
            .buf_arg(Args.entity_bone_tables, GPU.memory.get_buffer(ENTITY_BONE_TABLE))
            .buf_arg(Args.key_frames, GPU.memory.get_buffer(ANIM_KEY_FRAME))
            .buf_arg(Args.frame_times, GPU.memory.get_buffer(ANIM_FRAME_TIME))
            .buf_arg(Args.animation_timing_indices, GPU.memory.get_buffer(ANIM_TIMING_INDEX))
            .buf_arg(Args.animation_durations, GPU.memory.get_buffer(ANIM_DURATION))
            .buf_arg(Args.animation_tick_rates, GPU.memory.get_buffer(ANIM_TICK_RATE))
            .buf_arg(Args.entity_animation_layers, GPU.memory.get_buffer(ENTITY_ANIM_LAYER))
            .buf_arg(Args.entity_previous_layers, GPU.memory.get_buffer(ENTITY_PREV_LAYER))
            .buf_arg(Args.entity_animation_time, GPU.memory.get_buffer(ENTITY_ANIM_TIME))
            .buf_arg(Args.entity_previous_time, GPU.memory.get_buffer(ENTITY_PREV_TIME))
            .buf_arg(Args.entity_animation_blend, GPU.memory.get_buffer(ENTITY_ANIM_BLEND));
    }
}
