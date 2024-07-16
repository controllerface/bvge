package com.controllerface.bvge.cl.kernels.physics;

import com.controllerface.bvge.cl.kernels.GPUKernel;

public class CountCandidates_k extends GPUKernel
{
    public enum Args
    {
        bounds_bank_data,
        in_bounds,
        key_bank,
        key_counts,
        candidates,
        x_subdivisions,
        key_count_length,
        max_index,
    }

    public CountCandidates_k(long command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr, kernel_ptr);
    }
}
