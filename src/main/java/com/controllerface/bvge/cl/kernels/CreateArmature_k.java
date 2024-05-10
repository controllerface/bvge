package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPUKernel;

public class CreateArmature_k extends GPUKernel
{
    public enum Args
    {
        armatures,
        armature_root_hulls,
        armature_model_indices,
        armature_model_transforms,
        armature_flags,
        armature_hull_tables,
        armature_bone_tables,
        armature_masses,
        armature_animation_indices,
        armature_animation_elapsed,
        armature_motion_states,
        target,
        new_armature,
        new_armature_root_hull,
        new_armature_model_id,
        new_armature_model_transform,
        new_armature_flags,
        new_armature_hull_table,
        new_armature_bone_table,
        new_armature_mass,
        new_armature_animation_index,
        new_armature_animation_time,
        new_armature_animation_state;
    }

    public CreateArmature_k(long command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr, kernel_ptr);
    }
}
