package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.cl.GPUKernel;

public class CompleteCandidatesMultiBlockOut_k extends GPUKernel
{
    private static final GPU.Program program = GPU.Program.scan_key_candidates;
    private static final GPU.Kernel kernel = GPU.Kernel.complete_candidates_multi_block_out;

    public enum Args
    {
        input,
        output,
        sz,
        buffer,
        part,
        n;
    }

    public CompleteCandidatesMultiBlockOut_k(long command_queue_ptr)
    {
        super(command_queue_ptr, program.kernel_ptr(kernel));
    }
}
