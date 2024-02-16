package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.cl.GPUKernel;

public class AABBCollide_k extends GPUKernel
{
    private static final GPU.Program program = GPU.Program.aabb_collide;
    private static final GPU.Kernel kernel = GPU.Kernel.aabb_collide;

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

    public AABBCollide_k(long command_queue_ptr)
    {
        super(command_queue_ptr, program.kernel_ptr(kernel));
    }
}
