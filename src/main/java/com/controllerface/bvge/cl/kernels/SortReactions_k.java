package com.controllerface.bvge.cl.kernels;

public class SortReactions_k extends GPUKernel
{
    public enum Args
    {
        reactions_in,
        reactions_out,
        reaction_index,
        point_reactions,
        point_offsets;
    }

    public SortReactions_k(long command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr, kernel_ptr);
    }
}
