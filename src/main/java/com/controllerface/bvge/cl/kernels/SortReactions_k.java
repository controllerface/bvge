package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.cl.GPUKernel;
import com.controllerface.bvge.cl.GPUProgram;
import org.jocl.Pointer;
import org.jocl.Sizeof;
import org.jocl.cl_command_queue;

public class SortReactions_k extends GPUKernel<SortReactions_k.Args>
{
    public enum Args implements GPUKernelArg
    {
        reactions_in(Sizeof.cl_mem),
        reactions_out(Sizeof.cl_mem),
        reaction_index(Sizeof.cl_mem),
        point_reactions(Sizeof.cl_mem),
        point_offsets(Sizeof.cl_mem);

        public final long size;
        Args(long size) { this.size = size; }
        @Override public long size() { return size; }
    }

    public SortReactions_k(cl_command_queue command_queue, GPUProgram program)
    {
        super(command_queue, program.kernels().get(GPU.Kernel.sort_reactions), Args.values());
    }
}
