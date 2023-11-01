package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.cl.GPUKernel;
import com.controllerface.bvge.cl.GPUProgram;
import org.jocl.Pointer;
import org.jocl.Sizeof;
import org.jocl.cl_command_queue;

public class CompactEdges_k extends GPUKernel
{
    public CompactEdges_k(cl_command_queue command_queue, GPUProgram program)
    {
        super(command_queue, program.kernels().get(GPU.Kernel.compact_edges), 2);
        def_arg(0, Sizeof.cl_mem);
        def_arg(1, Sizeof.cl_mem);
    }

    public void set_edge_shift(Pointer edge_shift)
    {
        new_arg(0, Sizeof.cl_mem, edge_shift);
    }

    public void set_edges(Pointer edges)
    {
        new_arg(1, Sizeof.cl_mem, edges);
    }
}
