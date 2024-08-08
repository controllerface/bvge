package com.controllerface.bvge.gpu.cl.kernels.physics;

import com.controllerface.bvge.gpu.cl.contexts.CL_CommandQueue;
import com.controllerface.bvge.gpu.cl.kernels.GPUKernel;

public class SortReactions_k extends GPUKernel
{
    public enum Args
    {
        reactions_in,
        reactions_out,
        reaction_index,
        point_reactions,
        point_offsets,
        max_index,
    }

    public SortReactions_k(CL_CommandQueue command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr, kernel_ptr);
    }
}
