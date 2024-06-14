package com.controllerface.bvge.cl.kernels;

public class RootHullCount_k extends GPUKernel
{
    public enum Args
    {
        entity_model_indices,
        counter,
        model_id;
    }

    public RootHullCount_k(long command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr, kernel_ptr);
    }
}
