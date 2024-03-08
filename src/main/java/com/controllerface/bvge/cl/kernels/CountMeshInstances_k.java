package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.cl.GPUKernel;

public class CountMeshInstances_k extends GPUKernel
{
    private static final GPU.Program program = GPU.Program.mesh_query;
    private static final GPU.Kernel kernel = GPU.Kernel.count_mesh_instances;

    public enum Args
    {
        hull_mesh_ids,
        counters,
        query,
        total,
        count;
    }

    public CountMeshInstances_k(long command_queue_ptr)
    {
        super(command_queue_ptr, program.gpu.kernel_ptr(kernel));
    }
}
