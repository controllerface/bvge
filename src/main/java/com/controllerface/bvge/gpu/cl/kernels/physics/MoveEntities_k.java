package com.controllerface.bvge.gpu.cl.kernels.physics;

import com.controllerface.bvge.gpu.GPU;
import com.controllerface.bvge.gpu.cl.contexts.CL_CommandQueue;
import com.controllerface.bvge.gpu.cl.kernels.GPUKernel;
import com.controllerface.bvge.gpu.cl.kernels.KernelType;
import com.controllerface.bvge.gpu.cl.programs.GPUProgram;

import static com.controllerface.bvge.memory.types.CoreBufferType.*;

public class MoveEntities_k extends GPUKernel
{
    public enum Args
    {
        hulls,
        entities,
        entity_flags,
        entity_motion_states,
        entity_hull_tables,
        hull_point_tables,
        hull_integrity,
        hull_flags,
        point_flags,
        point_hit_counts,
        points,
        dt,
        max_entity,
    }

    public MoveEntities_k(CL_CommandQueue command_queue_ptr, GPUProgram program)
    {
        super(command_queue_ptr, program.get_kernel(KernelType.move_entities));
    }

    public GPUKernel init(float time_step)
    {
        return this.buf_arg(Args.hulls, GPU.memory.get_buffer(HULL))
            .buf_arg(Args.entities, GPU.memory.get_buffer(ENTITY))
            .buf_arg(Args.entity_flags, GPU.memory.get_buffer(ENTITY_FLAG))
            .buf_arg(Args.entity_motion_states, GPU.memory.get_buffer(ENTITY_MOTION_STATE))
            .buf_arg(Args.entity_hull_tables, GPU.memory.get_buffer(ENTITY_HULL_TABLE))
            .buf_arg(Args.hull_point_tables, GPU.memory.get_buffer(HULL_POINT_TABLE))
            .buf_arg(Args.hull_integrity, GPU.memory.get_buffer(HULL_INTEGRITY))
            .buf_arg(Args.hull_flags, GPU.memory.get_buffer(HULL_FLAG))
            .buf_arg(Args.point_flags, GPU.memory.get_buffer(POINT_FLAG))
            .buf_arg(Args.point_hit_counts, GPU.memory.get_buffer(POINT_HIT_COUNT))
            .buf_arg(Args.points, GPU.memory.get_buffer(POINT))
            .set_arg(Args.dt, time_step);
    }
}
