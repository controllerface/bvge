package com.controllerface.bvge.gpu.cl.kernels.crud;

import com.controllerface.bvge.gpu.cl.contexts.CL_CommandQueue;
import com.controllerface.bvge.gpu.cl.kernels.GPUKernel;
import com.controllerface.bvge.gpu.cl.kernels.KernelType;
import com.controllerface.bvge.gpu.cl.programs.GPUProgram;
import com.controllerface.bvge.memory.groups.CoreBufferGroup;

import static com.controllerface.bvge.memory.types.CoreBufferType.*;

public class UpdateBlockPosition_k extends GPUKernel
{
    public enum Args
    {
        entities,
        entity_root_hulls,
        hull_point_tables,
        points,
        target,
        new_value,
    }

    public UpdateBlockPosition_k(CL_CommandQueue command_queue_ptr, GPUProgram program)
    {
        super(command_queue_ptr, program.get_kernel(KernelType.update_block_position));
    }

    public GPUKernel init(CoreBufferGroup core_buffers)
    {
        return this.buf_arg(Args.entities, core_buffers.buffer(ENTITY))
            .buf_arg(Args.entity_root_hulls, core_buffers.buffer(ENTITY_ROOT_HULL))
            .buf_arg(Args.hull_point_tables, core_buffers.buffer(HULL_POINT_TABLE))
            .buf_arg(Args.points, core_buffers.buffer(POINT));
    }
}
