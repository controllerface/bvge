package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.cl.GPUKernel;

public class ApplyReactions_k extends GPUKernel
{
    private static final GPU.Program program = GPU.Program.sat_collide;
    private static final GPU.Kernel kernel = GPU.Kernel.apply_reactions;

    public enum Args
    {
        reactions,
        points,
        anti_gravity,
        point_reactions,
        point_offsets;
    }

    public ApplyReactions_k(long command_queue_ptr)
    {
        super(command_queue_ptr, program.kernel_ptr(kernel));
    }
}
