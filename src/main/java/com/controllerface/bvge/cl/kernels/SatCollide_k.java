package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.cl.GPUKernel;

public class SatCollide_k extends GPUKernel
{
    private static final GPU.Program program = GPU.Program.sat_collide;
    private static final GPU.Kernel kernel = GPU.Kernel.sat_collide;

    public enum Args
    {
        candidates,
        hulls,
        element_tables,
        hull_flags,
        vertex_tables,
        points,
        edges,
        reactions,
        reaction_index,
        point_reactions,
        masses,
        counter;
    }

    public SatCollide_k(long command_queue_ptr)
    {
        super(command_queue_ptr, program.kernel_ptr(kernel));
    }
}
