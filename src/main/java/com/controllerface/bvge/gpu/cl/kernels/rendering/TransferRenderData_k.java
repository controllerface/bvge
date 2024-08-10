package com.controllerface.bvge.gpu.cl.kernels.rendering;

import com.controllerface.bvge.gpu.GPU;
import com.controllerface.bvge.gpu.cl.buffers.CL_Buffer;
import com.controllerface.bvge.gpu.cl.contexts.CL_CommandQueue;
import com.controllerface.bvge.gpu.cl.kernels.GPUKernel;
import com.controllerface.bvge.gpu.cl.kernels.KernelType;
import com.controllerface.bvge.gpu.cl.programs.GPUProgram;

import static com.controllerface.bvge.memory.types.ReferenceBufferType.*;
import static com.controllerface.bvge.memory.types.RenderBufferType.*;

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

    public GPUKernel init(CL_Buffer command_buffer,
                          CL_Buffer element_buffer,
                          CL_Buffer vertex_buffer,
                          CL_Buffer uv_buffer,
                          CL_Buffer color_buffer,
                          CL_Buffer slot_buffer,
                          CL_Buffer mesh_transfer_buffer)
    {
        return this.buf_arg(Args.command_buffer, command_buffer)
            .buf_arg(Args.element_buffer, element_buffer)
            .buf_arg(Args.vertex_buffer, vertex_buffer)
            .buf_arg(Args.uv_buffer, uv_buffer)
            .buf_arg(Args.color_buffer, color_buffer)
            .buf_arg(Args.slot_buffer, slot_buffer)
            .buf_arg(Args.mesh_transfer, mesh_transfer_buffer)
            .buf_arg(Args.hull_point_tables, GPU.memory.get_buffer(RENDER_HULL_POINT_TABLE))
            .buf_arg(Args.hull_mesh_ids, GPU.memory.get_buffer(RENDER_HULL_MESH_ID))
            .buf_arg(Args.hull_entity_ids, GPU.memory.get_buffer(RENDER_HULL_ENTITY_ID))
            .buf_arg(Args.hull_flags, GPU.memory.get_buffer(RENDER_HULL_FLAG))
            .buf_arg(Args.hull_uv_offsets, GPU.memory.get_buffer(RENDER_HULL_UV_OFFSET))
            .buf_arg(Args.hull_integrity, GPU.memory.get_buffer(RENDER_HULL_INTEGRITY))
            .buf_arg(Args.entity_flags, GPU.memory.get_buffer(RENDER_ENTITY_FLAG))
            .buf_arg(Args.mesh_vertex_tables, GPU.memory.get_buffer(MESH_VERTEX_TABLE))
            .buf_arg(Args.mesh_face_tables, GPU.memory.get_buffer(MESH_FACE_TABLE))
            .buf_arg(Args.mesh_faces, GPU.memory.get_buffer(MESH_FACE))
            .buf_arg(Args.points, GPU.memory.get_buffer(RENDER_POINT))
            .buf_arg(Args.point_hit_counts, GPU.memory.get_buffer(RENDER_POINT_HIT_COUNT))
            .buf_arg(Args.point_vertex_references, GPU.memory.get_buffer(RENDER_POINT_VERTEX_REFERENCE))
            .buf_arg(Args.uv_tables, GPU.memory.get_buffer(VERTEX_UV_TABLE))
            .buf_arg(Args.texture_uvs, GPU.memory.get_buffer(VERTEX_TEXTURE_UV));
    }
}
