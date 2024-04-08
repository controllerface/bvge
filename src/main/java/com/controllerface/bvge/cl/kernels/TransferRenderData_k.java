package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPUKernel;

public class TransferRenderData_k extends GPUKernel
{
    public enum Args
    {
        hull_element_tables,
        hull_mesh_ids,
        mesh_references,
        mesh_faces,
        points,
        point_vertex_references,
        uv_tables,
        texture_uvs,
        command_buffer,
        vertex_buffer,
        uv_buffer,
        element_buffer,
        mesh_details,
        mesh_transfer,
        offset;
    }

    public TransferRenderData_k(long command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr, kernel_ptr);
    }
}
