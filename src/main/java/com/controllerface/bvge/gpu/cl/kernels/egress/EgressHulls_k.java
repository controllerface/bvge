package com.controllerface.bvge.gpu.cl.kernels.egress;

import com.controllerface.bvge.gpu.cl.buffers.ResizableBuffer;
import com.controllerface.bvge.gpu.cl.contexts.CL_CommandQueue;
import com.controllerface.bvge.gpu.cl.kernels.GPUKernel;
import com.controllerface.bvge.gpu.cl.kernels.KernelType;
import com.controllerface.bvge.gpu.cl.programs.GPUProgram;
import com.controllerface.bvge.memory.GPUCoreMemory;
import com.controllerface.bvge.memory.groups.UnorderedCoreBufferGroup;

import static com.controllerface.bvge.memory.types.CoreBufferType.*;

public class EgressHulls_k extends GPUKernel
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
        new_hulls,
        max_hull,
    }

    public EgressHulls_k(CL_CommandQueue command_queue_ptr, GPUProgram program)
    {
        super(command_queue_ptr, program.get_kernel(KernelType.egress_hulls));
    }

    public GPUKernel init(GPUCoreMemory core_memory,
                          UnorderedCoreBufferGroup sector_buffers,
                          ResizableBuffer b_hull_shift)
    {
        return this.buf_arg(Args.hulls_in, core_memory.get_buffer(HULL))
            .buf_arg(Args.hull_scales_in, core_memory.get_buffer(HULL_SCALE))
            .buf_arg(Args.hull_rotations_in, core_memory.get_buffer(HULL_ROTATION))
            .buf_arg(Args.hull_frictions_in, core_memory.get_buffer(HULL_FRICTION))
            .buf_arg(Args.hull_restitutions_in, core_memory.get_buffer(HULL_RESTITUTION))
            .buf_arg(Args.hull_point_tables_in, core_memory.get_buffer(HULL_POINT_TABLE))
            .buf_arg(Args.hull_edge_tables_in, core_memory.get_buffer(HULL_EDGE_TABLE))
            .buf_arg(Args.hull_bone_tables_in, core_memory.get_buffer(HULL_BONE_TABLE))
            .buf_arg(Args.hull_entity_ids_in, core_memory.get_buffer(HULL_ENTITY_ID))
            .buf_arg(Args.hull_flags_in, core_memory.get_buffer(HULL_FLAG))
            .buf_arg(Args.hull_mesh_ids_in, core_memory.get_buffer(HULL_MESH_ID))
            .buf_arg(Args.hull_uv_offsets_in, core_memory.get_buffer(HULL_UV_OFFSET))
            .buf_arg(Args.hull_integrity_in, core_memory.get_buffer(HULL_INTEGRITY))
            .buf_arg(Args.hulls_out, sector_buffers.buffer(HULL))
            .buf_arg(Args.hull_scales_out, sector_buffers.buffer(HULL_SCALE))
            .buf_arg(Args.hull_rotations_out, sector_buffers.buffer(HULL_ROTATION))
            .buf_arg(Args.hull_frictions_out, sector_buffers.buffer(HULL_FRICTION))
            .buf_arg(Args.hull_restitutions_out, sector_buffers.buffer(HULL_RESTITUTION))
            .buf_arg(Args.hull_point_tables_out, sector_buffers.buffer(HULL_POINT_TABLE))
            .buf_arg(Args.hull_edge_tables_out, sector_buffers.buffer(HULL_EDGE_TABLE))
            .buf_arg(Args.hull_bone_tables_out, sector_buffers.buffer(HULL_BONE_TABLE))
            .buf_arg(Args.hull_entity_ids_out, sector_buffers.buffer(HULL_ENTITY_ID))
            .buf_arg(Args.hull_flags_out, sector_buffers.buffer(HULL_FLAG))
            .buf_arg(Args.hull_mesh_ids_out, sector_buffers.buffer(HULL_MESH_ID))
            .buf_arg(Args.hull_uv_offsets_out, sector_buffers.buffer(HULL_UV_OFFSET))
            .buf_arg(Args.hull_integrity_out, sector_buffers.buffer(HULL_INTEGRITY))
            .buf_arg(Args.new_hulls, b_hull_shift);
    }
}
