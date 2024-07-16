package com.controllerface.bvge.cl.kernels.egress;

import com.controllerface.bvge.cl.kernels.GPUKernel;

public class EgressEntities_k extends GPUKernel
{
    public enum Args
    {
        point_hull_indices_in,
        point_bone_tables_in,
        edges_in,
        hull_point_tables_in,
        hull_edge_tables_in,
        hull_bone_tables_in,
        hull_bind_pose_indices_in,
        entity_bone_parent_ids_in,

        entities_in,
        entity_animation_time_in,
        entity_motion_states_in,
        entity_animation_layers_in,
        entity_animation_previous_in,
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
        entity_motion_states_out,
        entity_animation_layers_out,
        entity_animation_previous_out,
        entity_hull_tables_out,
        entity_bone_tables_out,
        entity_masses_out,
        entity_root_hulls_out,
        entity_model_indices_out,
        entity_model_transforms_out,
        entity_types_out,
        entity_flags_out,

        new_points,
        new_edges,
        new_hulls,
        new_hull_bones,
        new_entity_bones,
        counters,
        max_entity,
    }

    public EgressEntities_k(long command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr, kernel_ptr);
    }
}
