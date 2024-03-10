package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPUKernel;

public class PrepareEdges_k extends GPUKernel
{
    public enum Args
    {
        points,
        edges,
        vertex_vbo,
        flag_vbo,
        offset;
    }

    public PrepareEdges_k(long command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr,kernel_ptr);
    }
}
