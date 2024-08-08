package com.controllerface.bvge.gpu.cl.kernels.rendering;

import com.controllerface.bvge.gpu.cl.contexts.CL_CommandQueue;
import com.controllerface.bvge.gpu.cl.kernels.GPUKernel;

public class RootHullCount_k extends GPUKernel
{
    public enum Args
    {
        entity_model_indices,
        counter,
        model_id,
        max_entity,
    }

    public RootHullCount_k(CL_CommandQueue command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr, kernel_ptr);
    }
}
