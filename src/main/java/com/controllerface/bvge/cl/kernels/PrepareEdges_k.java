package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.cl.GPUKernel;
import com.controllerface.bvge.cl.GPUProgram;
import org.jocl.Pointer;
import org.jocl.Sizeof;
import org.jocl.cl_command_queue;

public class PrepareEdges_k extends GPUKernel
{
    public PrepareEdges_k(cl_command_queue command_queue, GPUProgram program)
    {
        super(command_queue, program.kernels().get(GPU.Kernel.prepare_edges), 5);
        def_arg(0, Sizeof.cl_mem);
        def_arg(1, Sizeof.cl_mem);
        def_arg(2, Sizeof.cl_mem);
        def_arg(3, Sizeof.cl_mem);
        def_arg(4, Sizeof.cl_int);
    }

    public void set_points(Pointer points)
    {
        new_arg(0, Sizeof.cl_mem, points);
    }

    public void set_edges(Pointer edges)
    {
        new_arg(1, Sizeof.cl_mem, edges);
    }
}
