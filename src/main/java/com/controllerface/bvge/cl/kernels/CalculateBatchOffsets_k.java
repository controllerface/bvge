package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.cl.GPUKernel;
import com.controllerface.bvge.cl.GPUProgram;
import org.jocl.Sizeof;
import org.jocl.cl_command_queue;

public class CalculateBatchOffsets_k extends GPUKernel
{
    public CalculateBatchOffsets_k(cl_command_queue command_queue, GPUProgram program)
    {
        super(command_queue, program.kernels().get(GPU.Kernel.calculate_batch_offsets), 3);
        int arg_index = 0;
        def_arg(arg_index++, Sizeof.cl_mem);
        def_arg(arg_index++, Sizeof.cl_mem);
        def_arg(arg_index++, Sizeof.cl_int);
        System.out.printf("set %d args for %s\n", arg_index, this.getClass().getSimpleName());
    }
}
