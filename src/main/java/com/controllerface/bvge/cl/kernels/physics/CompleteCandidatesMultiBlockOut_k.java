package com.controllerface.bvge.cl.kernels.physics;

import com.controllerface.bvge.cl.kernels.GPUKernel;

public class CompleteCandidatesMultiBlockOut_k extends GPUKernel
{
    public enum Args
    {
        input,
        output,
        sz,
        buffer,
        part,
        n;
    }

    public CompleteCandidatesMultiBlockOut_k(long command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr, kernel_ptr);
    }
}
