package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.cl.GPUKernel;

public class WriteMeshDetails_k extends GPUKernel
{
    private static final GPU.Program program = GPU.Program.mesh_query;
    private static final GPU.Kernel kernel = GPU.Kernel.write_mesh_details;

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

    public WriteMeshDetails_k(long command_queue_ptr)
    {
        super(command_queue_ptr, program.kernel_ptr(kernel));
    }
}
