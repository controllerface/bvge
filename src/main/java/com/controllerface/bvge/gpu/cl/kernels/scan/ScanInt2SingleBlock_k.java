package com.controllerface.bvge.gpu.cl.kernels.scan;

import com.controllerface.bvge.gpu.cl.contexts.CL_CommandQueue;
import com.controllerface.bvge.gpu.cl.kernels.GPUKernel;
import com.controllerface.bvge.gpu.cl.kernels.KernelType;
import com.controllerface.bvge.gpu.cl.programs.GPUProgram;

public class ScanInt2SingleBlock_k extends GPUKernel
{
    public enum Args
    {
        data,
        buffer,
        n;
    }

    public ScanInt2SingleBlock_k(CL_CommandQueue command_queue_ptr, GPUProgram program)
    {
        super(command_queue_ptr, program.get_kernel(KernelType.scan_int2_single_block));
    }
}
