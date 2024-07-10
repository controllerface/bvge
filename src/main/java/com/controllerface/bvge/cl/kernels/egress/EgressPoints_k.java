package com.controllerface.bvge.cl.kernels.egress;

import com.controllerface.bvge.cl.kernels.GPUKernel;

public class EgressPoints_k extends GPUKernel
{
    public enum Args
    {
        points_in,
        point_vertex_references_in,
        point_hull_indices_in,
        point_hit_counts_in,
        point_flags_in,
        point_bone_tables_in,
        points_out,
        point_vertex_references_out,
        point_hull_indices_out,
        point_hit_counts_out,
        point_flags_out,
        point_bone_tables_out,
        new_points,
        max_point,
    }

    public EgressPoints_k(long command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr, kernel_ptr);
    }
}
