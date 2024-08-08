package com.controllerface.bvge.gpu.cl.kernels.scan;

import com.controllerface.bvge.gpu.cl.kernels.GPUKernel;

public class ScanIntSingleBlockOut_k extends GPUKernel
{
    public enum Args
    {
        input,
        output,
        buffer,
        n;
    }

    public ScanIntSingleBlockOut_k(long command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr, kernel_ptr);
    }
}
