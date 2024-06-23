package com.controllerface.bvge.cl.kernels;

public class CompactHulls_k extends GPUKernel
{
    public enum Args
    {
        hull_shift,
        hulls,
        hull_scales,
        hull_mesh_ids,
        hull_uv_offsets,
        hull_rotations,
        hull_frictions,
        hull_restitutions,
        hull_integrity,
        hull_bone_tables,
        hull_entity_ids,
        hull_flags,
        hull_point_tables,
        hull_edge_tables,
        hull_aabb,
        hull_aabb_index,
        hull_aabb_key_table;
    }

    public CompactHulls_k(long command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr, kernel_ptr);
    }
}
