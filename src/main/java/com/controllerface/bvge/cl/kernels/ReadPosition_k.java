package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.cl.GPUKernel;
import com.controllerface.bvge.cl.GPUProgram;
import org.jocl.Pointer;
import org.jocl.Sizeof;
import org.jocl.cl_command_queue;

public class ReadPosition_k extends GPUKernel
{
    public ReadPosition_k(cl_command_queue command_queue, GPUProgram program)
    {
        super(command_queue, program.kernels().get(GPU.Kernel.read_position), 3);
        def_arg(0, Sizeof.cl_mem);
        def_arg(1, Sizeof.cl_float2);
        def_arg(2, Sizeof.cl_int);
    }

    public void set_armatures(Pointer armatures)
    {
        new_arg(0, Sizeof.cl_mem, armatures);
    }
}
