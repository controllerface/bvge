package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.cl.GPUKernel;

public class CountMeshBatches_k extends GPUKernel
{
    private static final GPU.Program program = GPU.Program.mesh_query;
    private static final GPU.Kernel kernel = GPU.Kernel.count_mesh_batches;

    public enum Args
    {
        mesh_details,
        total,
        max_per_batch,
        count;
    }

    public CountMeshBatches_k(long command_queue_ptr)
    {
        super(command_queue_ptr, program.kernel_ptr(kernel));
    }
}
