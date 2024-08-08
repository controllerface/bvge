package com.controllerface.bvge.gpu.cl.kernels.rendering;

import com.controllerface.bvge.gpu.cl.contexts.CL_CommandQueue;
import com.controllerface.bvge.gpu.cl.kernels.GPUKernel;
import com.controllerface.bvge.gpu.cl.kernels.KernelType;
import com.controllerface.bvge.gpu.cl.programs.GPUProgram;

public class TransferRenderData_k extends GPUKernel
{
    public enum Args
    {
        hull_point_tables,
        hull_mesh_ids,
        hull_entity_ids,
        hull_flags,
        hull_uv_offsets,
        hull_integrity,
        entity_flags,
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
        slot_buffer,
        element_buffer,
        mesh_details,
        mesh_texture,
        mesh_transfer,
        offset,
        max_index,
    }

    public TransferRenderData_k(CL_CommandQueue command_queue_ptr, GPUProgram program)
    {
        super(command_queue_ptr, program.get_kernel(KernelType.transfer_render_data));
    }
}
