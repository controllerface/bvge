package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPUKernel;

public class MoveEntities_k extends GPUKernel
{
    public enum Args
    {
        hulls,
        entities,
        entity_flags,
        entity_hull_tables,
        hull_point_tables,
        hull_flags,
        point_flags,
        point_hit_counts,
        points
    }

    public MoveEntities_k(long command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr, kernel_ptr);
    }
}
