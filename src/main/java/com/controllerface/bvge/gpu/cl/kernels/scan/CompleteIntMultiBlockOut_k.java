package com.controllerface.bvge.gpu.cl.kernels.scan;

import com.controllerface.bvge.gpu.cl.contexts.CL_CommandQueue;
import com.controllerface.bvge.gpu.cl.kernels.GPUKernel;

public class CompleteIntMultiBlockOut_k extends GPUKernel
{
    public enum Args
    {
        output,
        buffer,
        part,
        n;
    }

    public CompleteIntMultiBlockOut_k(CL_CommandQueue command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr, kernel_ptr);
    }
}
