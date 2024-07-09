package com.controllerface.bvge.cl.kernels;

public class MergeHull_k extends GPUKernel
{
    public enum Args
    {
        hulls_in,
        hull_scales_in,
        hull_rotations_in,
        hull_frictions_in,
        hull_restitutions_in,
        hull_point_tables_in,
        hull_edge_tables_in,
        hull_bone_tables_in,
        hull_entity_ids_in,
        hull_flags_in,
        hull_mesh_ids_in,
        hull_uv_offsets_in,
        hull_integrity_in,
        hulls_out,
        hull_scales_out,
        hull_rotations_out,
        hull_frictions_out,
        hull_restitutions_out,
        hull_point_tables_out,
        hull_edge_tables_out,
        hull_bone_tables_out,
        hull_entity_ids_out,
        hull_flags_out,
        hull_mesh_ids_out,
        hull_uv_offsets_out,
        hull_integrity_out,
        hull_offset,
        hull_bone_offset,
        entity_offset,
        edge_offset,
        point_offset,
        max_hull,
    }

    public MergeHull_k(long command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr, kernel_ptr);
    }
}
