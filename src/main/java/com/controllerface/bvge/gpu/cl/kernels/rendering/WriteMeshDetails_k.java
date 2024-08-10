package com.controllerface.bvge.gpu.cl.kernels.rendering;

import com.controllerface.bvge.gpu.GPU;
import com.controllerface.bvge.gpu.cl.buffers.CL_Buffer;
import com.controllerface.bvge.gpu.cl.contexts.CL_CommandQueue;
import com.controllerface.bvge.gpu.cl.kernels.GPUKernel;
import com.controllerface.bvge.gpu.cl.kernels.KernelType;
import com.controllerface.bvge.gpu.cl.programs.GPUProgram;

import static com.controllerface.bvge.memory.types.ReferenceBufferType.MESH_FACE_TABLE;
import static com.controllerface.bvge.memory.types.ReferenceBufferType.MESH_VERTEX_TABLE;
import static com.controllerface.bvge.memory.types.RenderBufferType.*;

public class WriteMeshDetails_k extends GPUKernel
{
    public enum Args
    {
        hull_mesh_ids,
        hull_flags,
        hull_entity_ids,
        entity_flags,
        mesh_vertex_tables,
        mesh_face_tables,
        counters,
        query,
        offsets,
        mesh_details,
        mesh_texture,
        count,
        max_hull,
    }

    public WriteMeshDetails_k(CL_CommandQueue command_queue_ptr, GPUProgram program)
    {
        super(command_queue_ptr, program.get_kernel(KernelType.write_mesh_details));
    }

    public GPUKernel init(CL_Buffer counter_buf,
                          CL_Buffer query_buf,
                          CL_Buffer offset_buf,
                          int mesh_count)
    {
        return this.buf_arg(Args.counters, counter_buf)
            .buf_arg(Args.query, query_buf)
            .buf_arg(Args.offsets, offset_buf)
            .set_arg(Args.count, mesh_count)
            .buf_arg(Args.hull_mesh_ids, GPU.memory.get_buffer(RENDER_HULL_MESH_ID))
            .buf_arg(Args.hull_flags, GPU.memory.get_buffer(RENDER_HULL_FLAG))
            .buf_arg(Args.hull_entity_ids, GPU.memory.get_buffer(RENDER_HULL_ENTITY_ID))
            .buf_arg(Args.entity_flags, GPU.memory.get_buffer(RENDER_ENTITY_FLAG))
            .buf_arg(Args.mesh_vertex_tables, GPU.memory.get_buffer(MESH_VERTEX_TABLE))
            .buf_arg(Args.mesh_face_tables, GPU.memory.get_buffer(MESH_FACE_TABLE));
    }
}
