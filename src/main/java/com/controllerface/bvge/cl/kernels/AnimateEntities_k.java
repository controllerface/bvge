package com.controllerface.bvge.cl.kernels;

public class AnimateEntities_k extends GPUKernel
{
    public enum Args
    {
        armature_bones,
        bone_bind_poses,
        model_transforms,
        entity_flags,
        armature_bone_reference_ids,
        armature_bone_parent_ids,
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
        entity_animation_indices,
        entity_animation_elapsed,
        entity_animation_blend,
        delta_time;
    }

    public AnimateEntities_k(long command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr, kernel_ptr);
    }
}
