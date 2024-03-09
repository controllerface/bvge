package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPUKernel;

public class AABBCollide_k extends GPUKernel
{
    public enum Args
    {
        bounds,
        bounds_bank_data,
        hull_flags,
        candidates,
        match_offsets,
        key_map,
        key_bank,
        key_counts,
        key_offsets,
        matches,
        used,
        counter,
        x_subdivisions,
        key_count_length;
    }

    public AABBCollide_k(long command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr, kernel_ptr);
    }
}
