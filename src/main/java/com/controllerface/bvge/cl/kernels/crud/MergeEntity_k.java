package com.controllerface.bvge.cl.kernels.crud;

import com.controllerface.bvge.cl.kernels.GPUKernel;

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

    public MergeEntity_k(long command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr, kernel_ptr);
    }
}