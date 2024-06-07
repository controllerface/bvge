package com.controllerface.bvge.cl.kernels;

public class CreateMeshReference_k extends GPUKernel
{
    public enum Args
    {
        mesh_vertex_tables,
        mesh_face_tables,
        target,
        new_mesh_vertex_table,
        new_mesh_face_table,
    }

    public CreateMeshReference_k(long command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr, kernel_ptr);
    }
}
