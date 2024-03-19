package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPUKernel;

public class CreateEdge_k extends GPUKernel
{
    public enum Args
    {
        edges,
        edge_lengths,
        edge_flags,
        target,
        new_edge,
        new_edge_length,
        new_edge_flag;
    }

    public CreateEdge_k(long command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr, kernel_ptr);
    }
}
