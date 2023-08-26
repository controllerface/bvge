package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.cl.GPUKernel;
import com.controllerface.bvge.cl.GPUProgram;
import org.jocl.Pointer;
import org.jocl.Sizeof;
import org.jocl.cl_command_queue;

public class SortReactions_k extends GPUKernel
{
    public SortReactions_k(cl_command_queue command_queue, GPUProgram program)
    {
        super(command_queue, program.kernels().get(GPU.Kernel.sort_reactions), 4);
        def_arg(0, Sizeof.cl_mem);
        def_arg(1, Sizeof.cl_mem);
        def_arg(2, Sizeof.cl_mem);
        def_arg(3, Sizeof.cl_mem);
    }

    public void set_reactions(Pointer reactions)
    {
        new_arg(2, Sizeof.cl_mem, reactions);
    }

    public void set_offsets(Pointer offsets)
    {
        new_arg(3, Sizeof.cl_mem, offsets);
    }
}
