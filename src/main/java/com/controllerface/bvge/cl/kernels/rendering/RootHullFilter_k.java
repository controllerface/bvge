package com.controllerface.bvge.cl.kernels.rendering;

import com.controllerface.bvge.cl.kernels.GPUKernel;

public class RootHullFilter_k extends GPUKernel
{
    public enum Args
    {
        entity_root_hulls,
        entity_model_indices,
        hulls_out,
        counter,
        model_id,
        max_entity,
    }

    public RootHullFilter_k(long command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr, kernel_ptr);
    }
}
