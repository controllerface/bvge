package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.cl.GPUKernel;

public class BuildKeyMap_k extends GPUKernel
{
    private static final GPU.Program program = GPU.Program.build_key_map;
    private static final GPU.Kernel kernel = GPU.Kernel.build_key_map;

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

    public BuildKeyMap_k(long command_queue_ptr)
    {
        super(command_queue_ptr, program.kernel_ptr(kernel));
    }
}
