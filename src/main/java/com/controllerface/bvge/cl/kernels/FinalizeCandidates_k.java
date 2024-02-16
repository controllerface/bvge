package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.cl.GPUKernel;

public class FinalizeCandidates_k extends GPUKernel
{
    private static final GPU.Program program = GPU.Program.locate_in_bounds;
    private static final GPU.Kernel kernel = GPU.Kernel.finalize_candidates;

    public enum Args
    {
        input_candidates,
        match_offsets,
        matches,
        used,
        counter,
        final_candidates;
    }

    public FinalizeCandidates_k(long command_queue_ptr)
    {
        super(command_queue_ptr, program.kernel_ptr(kernel));
    }
}
