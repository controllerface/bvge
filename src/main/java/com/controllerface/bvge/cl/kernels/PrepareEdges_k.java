package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.cl.GPUKernel;

public class PrepareEdges_k extends GPUKernel
{
    private static final GPU.Program program = GPU.Program.prepare_edges;
    private static final GPU.Kernel kernel = GPU.Kernel.prepare_edges;

    public enum Args
    {
        points,
        edges,
        vertex_vbo,
        flag_vbo,
        offset;
    }

    public PrepareEdges_k(long command_queue_ptr)
    {
        super(command_queue_ptr, program.kernel_ptr(kernel));
    }
}
