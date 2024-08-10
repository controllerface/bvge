package com.controllerface.bvge.gpu.cl.kernels.crud;

import com.controllerface.bvge.gpu.cl.contexts.CL_CommandQueue;
import com.controllerface.bvge.gpu.cl.kernels.GPUKernel;
import com.controllerface.bvge.gpu.cl.kernels.KernelType;
import com.controllerface.bvge.gpu.cl.programs.GPUProgram;
import com.controllerface.bvge.memory.GPUCoreMemory;
import com.controllerface.bvge.memory.groups.CoreBufferGroup;

import static com.controllerface.bvge.memory.types.CoreBufferType.*;

public class MergeHull_k extends GPUKernel
{
    public enum Args
    {
        hulls_in,
        hull_scales_in,
        hull_rotations_in,
        hull_frictions_in,
        hull_restitutions_in,
        hull_point_tables_in,
        hull_edge_tables_in,
        hull_bone_tables_in,
        hull_entity_ids_in,
        hull_flags_in,
        hull_mesh_ids_in,
        hull_uv_offsets_in,
        hull_integrity_in,
        hulls_out,
        hull_scales_out,
        hull_rotations_out,
        hull_frictions_out,
        hull_restitutions_out,
        hull_point_tables_out,
        hull_edge_tables_out,
        hull_bone_tables_out,
        hull_entity_ids_out,
        hull_flags_out,
        hull_mesh_ids_out,
        hull_uv_offsets_out,
        hull_integrity_out,
        hull_offset,
        hull_bone_offset,
        entity_offset,
        edge_offset,
        point_offset,
        max_hull,
    }

    public MergeHull_k(CL_CommandQueue command_queue_ptr, GPUProgram program)
    {
        super(command_queue_ptr, program.get_kernel(KernelType.merge_hull));
    }

    public GPUKernel init(GPUCoreMemory core_memory, CoreBufferGroup core_buffers)
    {
        return this.buf_arg(Args.hulls_in, core_buffers.buffer(HULL))
            .buf_arg(Args.hull_scales_in, core_buffers.buffer(HULL_SCALE))
            .buf_arg(Args.hull_rotations_in, core_buffers.buffer(HULL_ROTATION))
            .buf_arg(Args.hull_frictions_in, core_buffers.buffer(HULL_FRICTION))
            .buf_arg(Args.hull_restitutions_in, core_buffers.buffer(HULL_RESTITUTION))
            .buf_arg(Args.hull_point_tables_in, core_buffers.buffer(HULL_POINT_TABLE))
            .buf_arg(Args.hull_edge_tables_in, core_buffers.buffer(HULL_EDGE_TABLE))
            .buf_arg(Args.hull_bone_tables_in, core_buffers.buffer(HULL_BONE_TABLE))
            .buf_arg(Args.hull_entity_ids_in, core_buffers.buffer(HULL_ENTITY_ID))
            .buf_arg(Args.hull_flags_in, core_buffers.buffer(HULL_FLAG))
            .buf_arg(Args.hull_mesh_ids_in, core_buffers.buffer(HULL_MESH_ID))
            .buf_arg(Args.hull_uv_offsets_in, core_buffers.buffer(HULL_UV_OFFSET))
            .buf_arg(Args.hull_integrity_in, core_buffers.buffer(HULL_INTEGRITY))
            .buf_arg(Args.hulls_out, core_memory.get_buffer(HULL))
            .buf_arg(Args.hull_scales_out, core_memory.get_buffer(HULL_SCALE))
            .buf_arg(Args.hull_rotations_out, core_memory.get_buffer(HULL_ROTATION))
            .buf_arg(Args.hull_frictions_out, core_memory.get_buffer(HULL_FRICTION))
            .buf_arg(Args.hull_restitutions_out, core_memory.get_buffer(HULL_RESTITUTION))
            .buf_arg(Args.hull_point_tables_out, core_memory.get_buffer(HULL_POINT_TABLE))
            .buf_arg(Args.hull_edge_tables_out, core_memory.get_buffer(HULL_EDGE_TABLE))
            .buf_arg(Args.hull_bone_tables_out, core_memory.get_buffer(HULL_BONE_TABLE))
            .buf_arg(Args.hull_entity_ids_out, core_memory.get_buffer(HULL_ENTITY_ID))
            .buf_arg(Args.hull_flags_out, core_memory.get_buffer(HULL_FLAG))
            .buf_arg(Args.hull_mesh_ids_out, core_memory.get_buffer(HULL_MESH_ID))
            .buf_arg(Args.hull_uv_offsets_out, core_memory.get_buffer(HULL_UV_OFFSET))
            .buf_arg(Args.hull_integrity_out, core_memory.get_buffer(HULL_INTEGRITY));
    }
}
