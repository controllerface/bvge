package com.controllerface.bvge.gpu.cl.kernels.animation;

import com.controllerface.bvge.gpu.cl.contexts.CL_CommandQueue;
import com.controllerface.bvge.gpu.cl.kernels.GPUKernel;
import com.controllerface.bvge.gpu.cl.kernels.KernelType;
import com.controllerface.bvge.gpu.cl.programs.GPUProgram;

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
}
