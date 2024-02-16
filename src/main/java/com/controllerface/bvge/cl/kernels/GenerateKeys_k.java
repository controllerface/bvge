package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.cl.GPUKernel;

public class GenerateKeys_k extends GPUKernel
{
    private static final GPU.Program program = GPU.Program.generate_keys;
    private static final GPU.Kernel kernel = GPU.Kernel.generate_keys;

    public enum Args
    {
        bounds_index_data,
        bounds_bank_data,
        key_bank,
        key_counts,
        x_subdivisions,
        key_bank_length,
        key_count_length;
    }

    public GenerateKeys_k(long command_queue_ptr)
    {
        super(command_queue_ptr, program.kernel_ptr(kernel));
    }
}
