package com.controllerface.bvge.cl.kernels;

public class CountMeshBatches_k extends GPUKernel
{
    public enum Args
    {
        mesh_details,
        total,
        max_per_batch,
        count;
    }

    public CountMeshBatches_k(long command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr, kernel_ptr);
    }
}
