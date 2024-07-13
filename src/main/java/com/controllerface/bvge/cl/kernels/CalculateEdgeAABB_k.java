package com.controllerface.bvge.cl.kernels;

public class CalculateEdgeAABB_k extends GPUKernel
{
    public enum Args
    {
        edges,
        edge_flags,
        points,
        edge_aabb,
        edge_aabb_index,
        edge_aabb_key_table,
        args,
        max_edge,
    }

    public CalculateEdgeAABB_k(long command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr, kernel_ptr);
    }
}
