package com.controllerface.bvge.cl.kernels.egress;

import com.controllerface.bvge.cl.kernels.GPUKernel;

public class CountEgressEntities_k extends GPUKernel
{
    public enum Args
    {
        entity_flags,
        entity_hull_tables,
        entity_bone_tables,
        hull_flags,
        hull_point_tables,
        hull_edge_tables,
        hull_bone_tables,
        counters,
        max_entity,
    }

    public CountEgressEntities_k(long command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr, kernel_ptr);
    }
}