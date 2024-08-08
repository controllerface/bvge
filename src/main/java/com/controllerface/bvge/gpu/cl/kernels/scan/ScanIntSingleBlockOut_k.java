package com.controllerface.bvge.gpu.cl.kernels.scan;

import com.controllerface.bvge.gpu.cl.contexts.CL_CommandQueue;
import com.controllerface.bvge.gpu.cl.kernels.GPUKernel;
import com.controllerface.bvge.gpu.cl.kernels.KernelType;
import com.controllerface.bvge.gpu.cl.programs.GPUProgram;

public class ScanIntSingleBlockOut_k extends GPUKernel
{
    public enum Args
    {
        input,
        output,
        buffer,
        n;
    }

    public ScanIntSingleBlockOut_k(CL_CommandQueue command_queue_ptr, GPUProgram program)
    {
        super(command_queue_ptr, program.get_kernel(KernelType.scan_int_single_block_out));
    }
}
