package com.controllerface.bvge.cl.kernels;

public class ScanCandidatesMultiBlockOut_k extends GPUKernel
{
    public enum Args
    {
        input,
        output,
        buffer,
        part,
        n;
    }

    public ScanCandidatesMultiBlockOut_k(long command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr, kernel_ptr);
    }
}
