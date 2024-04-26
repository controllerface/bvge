package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPUKernel;

public class Integrate_k extends GPUKernel
{
    public enum Args
    {
        hulls,
        hull_scales,
        hull_point_tables,
        armature_accel,
        hull_rotations,
        points,
        point_hit_counts,
        point_flags,
        bounds,
        bounds_index_data,
        bounds_bank_data,
        hull_flags,
        hull_armature_ids,
        anti_gravity,
        args;
    }

    public Integrate_k(long command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr, kernel_ptr);
    }
}
