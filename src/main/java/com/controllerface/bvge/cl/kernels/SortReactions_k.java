package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.cl.GPUKernel;
import com.controllerface.bvge.cl.GPUProgram;
import org.jocl.Pointer;
import org.jocl.Sizeof;
import org.jocl.cl_command_queue;

public class SortReactions_k extends GPUKernel<SortReactions_k.Args>
{
    private static final GPU.Program program = GPU.Program.sat_collide;
    private static final GPU.Kernel kernel = GPU.Kernel.sort_reactions;

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

    public SortReactions_k(long command_queue_ptr)
    {
        super(command_queue_ptr, program.kernel_ptr(kernel), Args.values());
    }
}
