package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPUKernel;

public class CompactArmatures_k extends GPUKernel
{
    public enum Args
    {
        buffer_in_1,
        buffer_in_2,
        armatures,
        armature_masses,
        armature_root_hulls,
        armature_model_indices,
        armature_model_transforms,
        armature_flags,
        armature_animation_indices,
        armature_animation_elapsed,
        armature_animation_blend,
        armature_motion_states,
        armature_hull_tables,
        armature_bone_tables,
        hull_bone_tables,
        hull_armature_ids,
        hull_point_tables,
        hull_edge_tables,
        points,
        point_hull_indices,
        point_bone_tables,
        armature_bone_parent_ids,
        hull_bind_pose_indices,
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
