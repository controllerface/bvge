package com.controllerface.bvge.cl.kernels;

public class EgressBroken_k extends GPUKernel
{
    public enum Args
    {
        entities,
        entity_flags,
        entity_types,
        entity_model_ids,
        positions,
        types,
        model_ids,
        counter,
    }

    public EgressBroken_k(long command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr, kernel_ptr);
    }
}
