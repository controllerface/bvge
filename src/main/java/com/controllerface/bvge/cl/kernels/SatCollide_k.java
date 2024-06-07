package com.controllerface.bvge.cl.kernels;

public class SatCollide_k extends GPUKernel
{
    public enum Args
    {
        candidates,
        hulls,
        hull_scales,
        hull_frictions,
        hull_restitutions,
        hull_integrity,
        hull_point_tables,
        hull_edge_tables,
        hull_entity_ids,
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

    public SatCollide_k(long command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr, kernel_ptr);
    }
}
