package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.cl.GPUKernel;

public class AnimateArmatures_k extends GPUKernel
{
    private static final GPU.Program program = GPU.Program.animate_hulls;
    private static final GPU.Kernel kernel = GPU.Kernel.animate_armatures;

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
        armature_flags,
        hull_tables,
        key_frames,
        frame_times,
        animation_timing_indices,
        animation_timings,
        armature_animation_indices,
        armature_animation_elapsed,
        delta_time;
    }

    public AnimateArmatures_k(long command_queue_ptr)
    {
        super(command_queue_ptr, program.kernel_ptr(kernel));
    }
}
