package com.controllerface.bvge.cl.kernels.egress;

import com.controllerface.bvge.cl.kernels.GPUKernel;

public class EgressEdges_k extends GPUKernel
{
    public enum Args
    {
        edges_in,
        edge_lengths_in,
        edge_flags_in,
        edges_out,
        edge_lengths_out,
        edge_flags_out,
        new_edges,
        max_edge,
    }

    public EgressEdges_k(long command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr, kernel_ptr);
    }
}
