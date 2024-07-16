package com.controllerface.bvge.cl.kernels.physics;

import com.controllerface.bvge.cl.kernels.GPUKernel;

public class CalculateBatchOffsets_k extends GPUKernel
{
    public enum Args
    {
        mesh_offsets,
        mesh_details,
        count;
    }

    public CalculateBatchOffsets_k(long command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr, kernel_ptr);
    }
}
