package com.controllerface.bvge.cl.kernels.physics;

import com.controllerface.bvge.cl.kernels.GPUKernel;

public class ScanCandidatesSingleBlockOut_k extends GPUKernel
{
    public enum Args
    {
        input,
        output,
        sz,
        buffer,
        n;
    }

    public ScanCandidatesSingleBlockOut_k(long command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr, kernel_ptr);
    }
}
