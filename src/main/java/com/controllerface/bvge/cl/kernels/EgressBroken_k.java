package com.controllerface.bvge.cl.kernels;

public class EgressBroken_k extends GPUKernel
{
    public enum Args
    {
        entities,
        entity_flags,
        entity_hull_tables,
        entity_model_ids,
        hulls,
        hull_uv_offsets,
        positions,
        uv_offsets,
        model_ids,
        counter,
    }

    public EgressBroken_k(long command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr, kernel_ptr);
    }
}
