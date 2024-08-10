package com.controllerface.bvge.gpu.cl.kernels.physics;

import com.controllerface.bvge.gpu.GPU;
import com.controllerface.bvge.gpu.cl.buffers.BufferGroup;
import com.controllerface.bvge.gpu.cl.buffers.CL_Buffer;
import com.controllerface.bvge.gpu.cl.contexts.CL_CommandQueue;
import com.controllerface.bvge.gpu.cl.kernels.GPUKernel;
import com.controllerface.bvge.gpu.cl.kernels.KernelType;
import com.controllerface.bvge.gpu.cl.programs.GPUProgram;
import com.controllerface.bvge.memory.types.PhysicsBufferType;

import static com.controllerface.bvge.memory.types.CoreBufferType.*;
import static com.controllerface.bvge.memory.types.PhysicsBufferType.*;

public class SatCollide_k extends GPUKernel
{
    public enum Args
    {
        candidates,
        entity_model_transforms,
        entity_flags,
        hulls,
        hull_scales,
        hull_frictions,
        hull_restitutions,
        hull_integrity,
        hull_point_tables,
        hull_edge_tables,
        hull_entity_ids,
        hull_flags,
        point_flags,
        points,
        edges,
        edge_flags,
        reactions,
        reaction_index,
        point_reactions,
        masses,
        counter,
        dt,
        max_index,
    }

    public SatCollide_k(CL_CommandQueue command_queue_ptr, GPUProgram program)
    {
        super(command_queue_ptr, program.get_kernel(KernelType.sat_collide));
    }

    public GPUKernel init(BufferGroup<PhysicsBufferType> candidate_buffers,
                          BufferGroup<PhysicsBufferType> reaction_buffers,
                          CL_Buffer atomic_counter,
                          float time_step)
    {
        return this.buf_arg(Args.hulls, GPU.memory.get_buffer(HULL))
            .buf_arg(Args.hull_scales, GPU.memory.get_buffer(HULL_SCALE))
            .buf_arg(Args.hull_frictions, GPU.memory.get_buffer(HULL_FRICTION))
            .buf_arg(Args.hull_restitutions, GPU.memory.get_buffer(HULL_RESTITUTION))
            .buf_arg(Args.hull_integrity, GPU.memory.get_buffer(HULL_INTEGRITY))
            .buf_arg(Args.hull_point_tables, GPU.memory.get_buffer(HULL_POINT_TABLE))
            .buf_arg(Args.hull_edge_tables, GPU.memory.get_buffer(HULL_EDGE_TABLE))
            .buf_arg(Args.hull_entity_ids, GPU.memory.get_buffer(HULL_ENTITY_ID))
            .buf_arg(Args.hull_flags, GPU.memory.get_buffer(HULL_FLAG))
            .buf_arg(Args.point_flags, GPU.memory.get_buffer(POINT_FLAG))
            .buf_arg(Args.points, GPU.memory.get_buffer(POINT))
            .buf_arg(Args.edges, GPU.memory.get_buffer(EDGE))
            .buf_arg(Args.edge_flags, GPU.memory.get_buffer(EDGE_FLAG))
            .buf_arg(Args.masses, GPU.memory.get_buffer(ENTITY_MASS))
            .buf_arg(Args.entity_model_transforms, GPU.memory.get_buffer(ENTITY_TRANSFORM_ID))
            .buf_arg(Args.entity_flags, GPU.memory.get_buffer(ENTITY_FLAG))
            .buf_arg(Args.candidates, candidate_buffers.buffer(CANDIDATES))
            .buf_arg(Args.reactions, reaction_buffers.buffer(REACTIONS_IN))
            .buf_arg(Args.reaction_index, reaction_buffers.buffer(REACTION_INDEX))
            .buf_arg(Args.point_reactions, reaction_buffers.buffer(POINT_REACTION_COUNTS))
            .buf_arg(Args.counter, atomic_counter)
            .set_arg(Args.dt, time_step);
    }
}
