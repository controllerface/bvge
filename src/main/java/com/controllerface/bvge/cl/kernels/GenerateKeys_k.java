package com.controllerface.bvge.cl.kernels;

public class GenerateKeys_k extends GPUKernel
{
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

    public GenerateKeys_k(long command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr, kernel_ptr);
    }
}
