package com.controllerface.bvge.cl.kernels;

public class MergeEdge_k extends GPUKernel
{
    public enum Args
    {
        edges_in,
        edge_lengths_in,
        edge_flags_in,
        edges_out,
        edge_lengths_out,
        edge_flags_out,
        edge_offset,
        point_offset,
    }

    public MergeEdge_k(long command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr, kernel_ptr);
    }
}
