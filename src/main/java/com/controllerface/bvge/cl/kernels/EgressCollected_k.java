package com.controllerface.bvge.cl.kernels;

public class EgressCollected_k extends GPUKernel
{
    public enum Args
    {
        entity_flags,
        entity_types,
        entity_hull_tables,
        hull_flags,
        hull_uv_offsets,
        uv_offsets,
        flags,
        types,
        counter,
    }

    public EgressCollected_k(long command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr, kernel_ptr);
    }
}
