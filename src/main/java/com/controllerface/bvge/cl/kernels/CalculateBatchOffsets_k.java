package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.cl.GPUKernel;

public class CalculateBatchOffsets_k extends GPUKernel
{
    private static final GPU.Program program = GPU.Program.mesh_query;
    private static final GPU.Kernel kernel = GPU.Kernel.calculate_batch_offsets;

    public enum Args
    {
        mesh_offsets,
        mesh_details,
        count;
    }

    public CalculateBatchOffsets_k(long command_queue_ptr)
    {
        super(command_queue_ptr, program.gpu.kernel_ptr(kernel));
    }
}
