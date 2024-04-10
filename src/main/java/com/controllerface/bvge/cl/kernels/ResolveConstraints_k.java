package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPUKernel;

public class ResolveConstraints_k extends GPUKernel
{
    public enum Args
    {
        hull_edge_tables,
        bounds_bank_data,
        point,
        edges,
        edge_lengths,
        process_all;
    }

    public ResolveConstraints_k(long command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr, kernel_ptr);
    }
}
