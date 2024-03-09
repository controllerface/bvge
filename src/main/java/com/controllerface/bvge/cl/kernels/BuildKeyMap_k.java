package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPUKernel;

public class BuildKeyMap_k extends GPUKernel
{
    public enum Args
    {
        bounds_index_data,
        bounds_bank_data,
        key_map,
        key_offsets,
        key_counts,
        x_subdivisions,
        key_count_length;
    }

    public BuildKeyMap_k(long command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr, kernel_ptr);
    }
}
