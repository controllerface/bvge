package com.controllerface.bvge.gpu.cl.kernels.animation;

import com.controllerface.bvge.gpu.cl.kernels.GPUKernel;

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

    public AnimateEntities_k(long command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr, kernel_ptr);
    }
}
