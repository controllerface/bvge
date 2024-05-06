package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPUKernel;

public class TransferRenderData_k extends GPUKernel
{
    public enum Args
    {
        hull_point_tables,
        hull_mesh_ids,
        hull_armature_ids,
        hull_flags,
        armature_flags,
        mesh_vertex_tables,
        mesh_face_tables,
        mesh_faces,
        points,
        point_hit_counts,
        point_vertex_references,
        uv_tables,
        texture_uvs,
        command_buffer,
        vertex_buffer,
        uv_buffer,
        color_buffer,
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
