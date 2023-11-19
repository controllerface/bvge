package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.cl.GPUKernel;
import com.controllerface.bvge.cl.GPUProgram;
import org.jocl.Pointer;
import org.jocl.Sizeof;
import org.jocl.cl_command_queue;

public class PrepareTransforms_k extends GPUKernel
{
    public PrepareTransforms_k(cl_command_queue command_queue, GPUProgram program)
    {
        super(command_queue, program.kernels().get(GPU.Kernel.prepare_transforms), 5);
        def_arg(0, Sizeof.cl_mem);
        def_arg(1, Sizeof.cl_mem);
        def_arg(2, Sizeof.cl_mem);
        def_arg(3, Sizeof.cl_mem);
        def_arg(4, Sizeof.cl_int);
    }

    public void set_hulls(Pointer hulls)
    {
        new_arg(0, Sizeof.cl_mem, hulls);
    }

    public void set_rotations(Pointer rotations)
    {
        new_arg(1, Sizeof.cl_mem, rotations);
    }
}
