package com.controllerface.bvge.gpu.cl.kernels.scan;

import com.controllerface.bvge.gpu.cl.contexts.CL_CommandQueue;
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

    public ScanIntSingleBlockOut_k(CL_CommandQueue command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr, kernel_ptr);
    }
}
