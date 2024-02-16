package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.cl.GPUKernel;

public class SortReactions_k extends GPUKernel
{
    private static final GPU.Program program = GPU.Program.sat_collide;
    private static final GPU.Kernel kernel = GPU.Kernel.sort_reactions;

    public enum Args
    {
        reactions_in,
        reactions_out,
        reaction_index,
        point_reactions,
        point_offsets;
    }

    public SortReactions_k(long command_queue_ptr)
    {
        super(command_queue_ptr, program.kernel_ptr(kernel));
    }
}
