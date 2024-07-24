package com.controllerface.bvge.cl.kernels.physics;

import com.controllerface.bvge.cl.kernels.GPUKernel;

public class ResolveConstraints_k extends GPUKernel
{
    public enum Args
    {
        hulls,
        entities,
        hull_edge_tables,
        bounds_bank_data,
        point,
        edges,
        edge_lengths,
        edge_flags,
        edge_pins,
        process_all,
        max_hull,
    }

    public ResolveConstraints_k(long command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr, kernel_ptr);
    }
}
