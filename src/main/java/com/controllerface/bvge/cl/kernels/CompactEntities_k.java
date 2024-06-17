package com.controllerface.bvge.cl.kernels;

public class CompactEntities_k extends GPUKernel
{
    public enum Args
    {
        buffer_in_1,
        buffer_in_2,
        entities,
        entity_masses,
        entity_root_hulls,
        entity_model_indices,
        entity_model_transforms,
        entity_types,
        entity_flags,
        entity_animation_indices,
        entity_animation_elapsed,
        entity_animation_blend,
        entity_motion_states,
        entity_entity_hull_tables,
        entity_bone_tables,
        hull_bone_tables,
        hull_entity_ids,
        hull_point_tables,
        hull_edge_tables,
        points,
        point_hull_indices,
        point_bone_tables,
        entity_bone_parent_ids,
        hull_bind_pose_indices,
        edges,
        hull_bone_shift,
        point_shift,
        edge_shift,
        hull_shift,
        entity_bone_shift;
    }

    public CompactEntities_k(long command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr, kernel_ptr);
    }
}
