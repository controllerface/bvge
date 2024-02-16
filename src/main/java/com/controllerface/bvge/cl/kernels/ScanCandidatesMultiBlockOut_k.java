package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.cl.GPUKernel;

public class ScanCandidatesMultiBlockOut_k extends GPUKernel
{
    private static final GPU.Program program = GPU.Program.scan_key_candidates;
    private static final GPU.Kernel kernel = GPU.Kernel.scan_candidates_multi_block_out;

    public enum Args
    {
        input,
        output,
        buffer,
        part,
        n;
    }

    public ScanCandidatesMultiBlockOut_k(long command_queue_ptr)
    {
        super(command_queue_ptr, program.kernel_ptr(kernel));
    }
}
