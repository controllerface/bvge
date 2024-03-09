package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPUKernel;

public class WriteMeshDetails_k extends GPUKernel
{
    public enum Args
    {
        hull_mesh_ids,
        mesh_references,
        counters,
        query,
        offsets,
        mesh_details,
        count;
    }

    public WriteMeshDetails_k(long command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr, kernel_ptr);
    }
}
