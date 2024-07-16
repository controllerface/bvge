package com.controllerface.bvge.cl.kernels.rendering;

import com.controllerface.bvge.cl.kernels.GPUKernel;

public class PrepareEdges_k extends GPUKernel
{
    public enum Args
    {
        points,
        edges,
        edge_flags,
        vertex_vbo,
        flag_vbo,
        offset,
        max_edge,
    }

    public PrepareEdges_k(long command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr,kernel_ptr);
    }
}
