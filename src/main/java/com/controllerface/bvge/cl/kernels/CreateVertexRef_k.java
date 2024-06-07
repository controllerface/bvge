package com.controllerface.bvge.cl.kernels;

public class CreateVertexRef_k extends GPUKernel
{
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

    public CreateVertexRef_k(long command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr, kernel_ptr);
    }
}
