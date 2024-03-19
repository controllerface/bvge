package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPUKernel;

public class CompactEdges_k extends GPUKernel
{
    public enum Args
    {
        edge_shift,
        edges,
        edge_lengths,
        edge_flags;
    }

    public CompactEdges_k(long command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr, kernel_ptr);
    }
}
