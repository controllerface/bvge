package com.controllerface.bvge.gpu.cl.kernels.physics;

import com.controllerface.bvge.gpu.cl.contexts.CL_CommandQueue;
import com.controllerface.bvge.gpu.cl.kernels.GPUKernel;

public class BuildKeyBank_k extends GPUKernel
{
    public enum Args
    {
        hull_aabb_index,
        hull_aabb_key_table,
        key_bank,
        key_counts,
        x_subdivisions,
        key_bank_length,
        key_count_length,
        max_hull,
    }

    public BuildKeyBank_k(CL_CommandQueue command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr, kernel_ptr);
    }
}
