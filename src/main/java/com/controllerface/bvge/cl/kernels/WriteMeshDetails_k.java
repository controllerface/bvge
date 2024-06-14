package com.controllerface.bvge.cl.kernels;

public class WriteMeshDetails_k extends GPUKernel
{
    public enum Args
    {
        hull_mesh_ids,
        hull_flags,
        mesh_vertex_tables,
        mesh_face_tables,
        counters,
        query,
        offsets,
        mesh_details,
        mesh_texture,
        count;
    }

    public WriteMeshDetails_k(long command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr, kernel_ptr);
    }
}
