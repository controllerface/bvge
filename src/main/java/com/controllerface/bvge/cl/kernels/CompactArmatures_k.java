package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPUKernel;

public class CompactArmatures_k extends GPUKernel
{
    public enum Args
    {
        buffer_in_1,
        buffer_in_2,
        armatures,
        armature_root_hulls,
        armature_model_indices,
        armature_model_transforms,
        armature_flags,
        armature_animation_indices,
        armature_animation_elapsed,
        hull_tables,
        hulls,
        hull_bone_tables,
        hull_armature_ids,
        element_tables,
        points,
        point_hull_indices,
        bone_tables,
        bone_bind_tables,
        hull_bind_pose_indicies,
        edges,
        bone_shift,
        point_shift,
        edge_shift,
        hull_shift,
        bone_bind_shift;
    }

    public CompactArmatures_k(long command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr, kernel_ptr);
    }
}
