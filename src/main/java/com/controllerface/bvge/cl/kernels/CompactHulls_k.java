package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPUKernel;

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
        bone_tables,
        armature_ids,
        hull_flags,
        hull_point_tables,
        hull_edge_tables,
        hull_integrity_,
        bounds,
        bounds_index_data,
        bounds_bank_data;
    }

    public CompactHulls_k(long command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr, kernel_ptr);
    }
}
