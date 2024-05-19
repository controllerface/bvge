package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPUKernel;

public class CreateHull_k extends GPUKernel
{
    public enum Args
    {
        hulls,
        hull_scales,
        hull_rotations,
        hull_frictions,
        hull_restitutions,
        hull_point_tables,
        hull_edge_tables,
        hull_bone_tables,
        hull_armature_ids,
        hull_flags,
        hull_mesh_ids,
        hull_uv_offsets,
        hull_integrity,
        target,
        new_hull,
        new_hull_scale,
        new_rotation,
        new_friction,
        new_restitution,
        new_point_table,
        new_edge_table,
        new_bone_table,
        new_armature_id,
        new_flags,
        new_hull_mesh_id,
        new_hull_uv_offset,
        new_hull_integrity
    }

    public CreateHull_k(long command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr, kernel_ptr);
    }
}
