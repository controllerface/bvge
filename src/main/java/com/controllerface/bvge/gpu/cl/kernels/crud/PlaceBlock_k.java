package com.controllerface.bvge.gpu.cl.kernels.crud;

import com.controllerface.bvge.gpu.cl.contexts.CL_CommandQueue;
import com.controllerface.bvge.gpu.cl.kernels.GPUKernel;
import com.controllerface.bvge.gpu.cl.kernels.KernelType;
import com.controllerface.bvge.gpu.cl.programs.GPUProgram;
import com.controllerface.bvge.memory.groups.CoreBufferGroup;

import static com.controllerface.bvge.memory.types.CoreBufferType.*;

public class PlaceBlock_k extends GPUKernel
{
    public enum Args
    {
        entities,
        entity_hull_tables,
        hulls,
        hull_point_tables,
        hull_rotations,
        points,
        src,
        dest,
    }

    public PlaceBlock_k(CL_CommandQueue command_queue_ptr, GPUProgram program)
    {
        super(command_queue_ptr, program.get_kernel(KernelType.place_block));
    }

    public GPUKernel init(CoreBufferGroup core_buffers)
    {
        return this.buf_arg(Args.entities, core_buffers.buffer(ENTITY))
            .buf_arg(Args.entity_hull_tables, core_buffers.buffer(ENTITY_HULL_TABLE))
            .buf_arg(Args.hulls, core_buffers.buffer(HULL))
            .buf_arg(Args.hull_point_tables, core_buffers.buffer(HULL_POINT_TABLE))
            .buf_arg(Args.hull_rotations, core_buffers.buffer(HULL_ROTATION))
            .buf_arg(Args.points, core_buffers.buffer(POINT));
    }
}
