package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPUKernel;

public class AnimateArmatures_k extends GPUKernel
{
    public enum Args
    {
        armature_bones,
        bone_bind_poses,
        model_transforms,
        bone_bind_tables,
        bone_channel_tables,
        bone_pos_channel_tables,
        bone_rot_channel_tables,
        bone_scl_channel_tables,
        armature_model_transforms,
        hull_tables,
        key_frames,
        frame_times,
        animation_timing_indices,
        animation_durations,
        animation_tick_rates,
        armature_animation_indices,
        armature_animation_elapsed,
        delta_time;
    }

    public AnimateArmatures_k(long command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr, kernel_ptr);
    }
}
