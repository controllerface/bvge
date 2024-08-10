package com.controllerface.bvge.gpu.cl.kernels.egress;

import com.controllerface.bvge.gpu.cl.buffers.CL_Buffer;
import com.controllerface.bvge.gpu.cl.contexts.CL_CommandQueue;
import com.controllerface.bvge.gpu.cl.kernels.GPUKernel;
import com.controllerface.bvge.gpu.cl.kernels.KernelType;
import com.controllerface.bvge.gpu.cl.programs.GPUProgram;
import com.controllerface.bvge.memory.groups.CoreBufferGroup;

import static com.controllerface.bvge.memory.types.CoreBufferType.*;

public class CountEgressEntities_k extends GPUKernel
{
    public enum Args
    {
        entity_flags,
        entity_hull_tables,
        entity_bone_tables,
        hull_flags,
        hull_point_tables,
        hull_edge_tables,
        hull_bone_tables,
        counters,
        max_entity,
    }

    public CountEgressEntities_k(CL_CommandQueue command_queue_ptr, GPUProgram program)
    {
        super(command_queue_ptr, program.get_kernel(KernelType.count_egress_entities));
    }

    public GPUKernel init(CoreBufferGroup core_buffers, CL_Buffer egress_sizes)
    {
        return this.buf_arg(Args.entity_flags, core_buffers.buffer(ENTITY_FLAG))
            .buf_arg(Args.entity_hull_tables, core_buffers.buffer(ENTITY_HULL_TABLE))
            .buf_arg(Args.entity_bone_tables, core_buffers.buffer(ENTITY_BONE_TABLE))
            .buf_arg(Args.hull_flags, core_buffers.buffer(HULL_FLAG))
            .buf_arg(Args.hull_point_tables, core_buffers.buffer(HULL_POINT_TABLE))
            .buf_arg(Args.hull_edge_tables, core_buffers.buffer(HULL_EDGE_TABLE))
            .buf_arg(Args.hull_bone_tables, core_buffers.buffer(HULL_BONE_TABLE))
            .buf_arg(Args.counters, egress_sizes);
    }
}
