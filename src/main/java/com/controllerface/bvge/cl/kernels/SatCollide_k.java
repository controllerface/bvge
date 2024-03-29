package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPUKernel;

public class SatCollide_k extends GPUKernel
{
    public enum Args
    {
        candidates,
        hulls,
        hull_frictions,
        element_tables,
        hull_flags,
        vertex_tables,
        points,
        edges,
        edge_flags,
        reactions,
        reactions2,
        reaction_index,
        point_reactions,
        masses,
        counter,
        dt,
    }

    public SatCollide_k(long command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr, kernel_ptr);
    }
}
