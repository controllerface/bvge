package com.controllerface.bvge.cl.kernels;

public class CreatePoint_k extends GPUKernel
{
    public enum Args
    {
        points,
        point_vertex_references,
        point_hull_indices,
        point_hit_counts,
        point_flags,
        point_bone_tables,
        target,
        new_point,
        new_point_vertex_reference,
        new_point_hull_index,
        new_point_hit_count,
        new_point_flags,
        new_point_bone_table;
    }

    public CreatePoint_k(long command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr, kernel_ptr);
    }
}
