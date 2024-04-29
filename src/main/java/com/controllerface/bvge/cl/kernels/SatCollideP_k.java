package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPUKernel;

public class SatCollideP_k extends GPUKernel
{
    public enum Args
    {
        candidates,
        hulls,
        hull_scales,
        hull_frictions,
        hull_restitutions,
        hull_point_tables,
        hull_edge_tables,
        hull_armature_ids,
        hull_flags,
        point_flags,
        points,
        edges,
        edge_flags,
        reactions,
        reaction_index,
        point_reactions,
        masses,
        counter,
        dt,
    }

    public SatCollideP_k(long command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr, kernel_ptr);
    }
}
