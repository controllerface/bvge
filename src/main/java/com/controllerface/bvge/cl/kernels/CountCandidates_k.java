package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.cl.GPUKernel;

public class CountCandidates_k extends GPUKernel
{
    private static final GPU.Program program = GPU.Program.locate_in_bounds;
    private static final GPU.Kernel kernel = GPU.Kernel.count_candidates;

    public enum Args
    {
        bounds_bank_data,
        in_bounds,
        key_bank,
        key_counts,
        candidates,
        x_subdivisions,
        key_count_length;
    }

    public CountCandidates_k(long command_queue_ptr)
    {
        super(command_queue_ptr, program.gpu.kernel_ptr(kernel));
    }
}
