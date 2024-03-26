package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPUKernel;

public class ApplyReactions_k extends GPUKernel
{
    public enum Args
    {
        reactions,
        reactions2,
        points,
        anti_gravity,
        point_reactions,
        point_offsets;
    }

    public ApplyReactions_k(long command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr, kernel_ptr);
    }
}
