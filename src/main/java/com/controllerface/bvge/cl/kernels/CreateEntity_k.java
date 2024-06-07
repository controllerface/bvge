package com.controllerface.bvge.cl.kernels;

public class CreateEntity_k extends GPUKernel
{
    public enum Args
    {
        entities,
        entity_animation_elapsed,
        entity_motion_states,
        entity_animation_indices,
        entity_hull_tables,
        entity_bone_tables,
        entity_masses,
        entity_root_hulls,
        entity_model_indices,
        entity_model_transforms,
        entity_flags,
        target,
        new_entity,
        new_entity_animation_time,
        new_entity_animation_state,
        new_entity_animation_index,
        new_entity_hull_table,
        new_entity_bone_table,
        new_entity_mass,
        new_entity_root_hull,
        new_entity_model_id,
        new_entity_model_transform,
        new_entity_flags,

    }

    public CreateEntity_k(long command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr, kernel_ptr);
    }
}
