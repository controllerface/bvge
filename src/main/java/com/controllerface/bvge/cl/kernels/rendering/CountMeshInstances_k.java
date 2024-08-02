package com.controllerface.bvge.cl.kernels.rendering;

import com.controllerface.bvge.cl.kernels.GPUKernel;

public class CountMeshInstances_k extends GPUKernel
{
    public enum Args
    {
        hull_mesh_ids,
        hull_flags,
        hull_entity_ids,
        entity_flags,
        counters,
        query,
        total,
        count,
        max_hull,
    }

    public CountMeshInstances_k(long command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr, kernel_ptr);
    }
}