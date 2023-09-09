package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.cl.GPUKernel;
import com.controllerface.bvge.cl.GPUProgram;
import org.jocl.Sizeof;
import org.jocl.cl_command_queue;

public class ScanIntSingleBlock_k extends GPUKernel
{
    public ScanIntSingleBlock_k(cl_command_queue command_queue, GPUProgram program)
    {
        super(command_queue, program.kernels().get(GPU.Kernel.scan_int_single_block), 3);
        def_arg(0, Sizeof.cl_mem);
        def_arg(1, -1);
        def_arg(2, Sizeof.cl_int);
    }
}
