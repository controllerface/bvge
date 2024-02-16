package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.cl.GPUKernel;

public class TransferRenderData_k extends GPUKernel
{
    private static final GPU.Program program = GPU.Program.mesh_query;
    private static final GPU.Kernel kernel = GPU.Kernel.transfer_render_data;

    public enum Args
    {
        hull_element_tables,
        hull_mesh_ids,
        mesh_references,
        mesh_faces,
        points,
        vertex_tables,
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

    public TransferRenderData_k(long command_queue_ptr)
    {
        super(command_queue_ptr, program.kernel_ptr(kernel));
    }
}
