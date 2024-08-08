package com.controllerface.bvge.gpu.cl.kernels.physics;

import com.controllerface.bvge.gpu.cl.kernels.GPUKernel;

public class Integrate_k extends GPUKernel
{
    public enum Args
    {
        hull_point_tables,
        entity_accel,
        points,
        point_hit_counts,
        point_flags,
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
