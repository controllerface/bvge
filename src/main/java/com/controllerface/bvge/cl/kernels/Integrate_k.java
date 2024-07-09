package com.controllerface.bvge.cl.kernels;

public class Integrate_k extends GPUKernel
{
    public enum Args
    {
        hulls,
        hull_scales,
        hull_point_tables,
        entity_accel,
        hull_rotations,
        points,
        point_hit_counts,
        point_flags,
        bounds,
        bounds_index_data,
        bounds_bank_data,
        hull_flags,
        hull_entity_ids,
        anti_gravity,
        args,
        max_hull,
    }

    public Integrate_k(long command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr, kernel_ptr);
    }
}
