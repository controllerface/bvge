package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.cl.GPUKernel;

public class CreateVertexRef_k extends GPUKernel
{
    private static final GPU.Program program = GPU.Program.gpu_crud;
    private static final GPU.Kernel kernel = GPU.Kernel.create_vertex_reference;

    public enum Args
    {
        vertex_references,
        vertex_weights,
        uv_tables,
        target,
        new_vertex_reference,
        new_vertex_weights,
        new_uv_table;
    }

    public CreateVertexRef_k(long command_queue_ptr)
    {
        super(command_queue_ptr, program.gpu.kernel_ptr(kernel));
    }
}
