package com.controllerface.bvge.gpu.cl.kernels.physics;

import com.controllerface.bvge.gpu.GPU;
import com.controllerface.bvge.gpu.cl.buffers.BufferGroup;
import com.controllerface.bvge.gpu.cl.contexts.CL_CommandQueue;
import com.controllerface.bvge.gpu.cl.kernels.GPUKernel;
import com.controllerface.bvge.gpu.cl.kernels.KernelType;
import com.controllerface.bvge.gpu.cl.programs.GPUProgram;
import com.controllerface.bvge.memory.types.PhysicsBufferType;

import static com.controllerface.bvge.memory.types.CoreBufferType.*;
import static com.controllerface.bvge.memory.types.PhysicsBufferType.*;

public class ApplyReactions_k extends GPUKernel
{
    public enum Args
    {
        reactions,
        points,
        anti_gravity,
        point_flags,
        point_hit_counts,
        point_reactions,
        point_offsets,
        point_hull_indices,
        hull_flags,
        max_point,
    }

    public ApplyReactions_k(CL_CommandQueue command_queue_ptr, GPUProgram program)
    {
        super(command_queue_ptr, program.get_kernel(KernelType.apply_reactions));
    }

    public GPUKernel init(BufferGroup<PhysicsBufferType> reaction_buffers)
    {
        return this.buf_arg(Args.reactions, reaction_buffers.buffer(REACTIONS_OUT))
            .buf_arg(Args.point_reactions, reaction_buffers.buffer(POINT_REACTION_COUNTS))
            .buf_arg(Args.point_offsets, reaction_buffers.buffer(POINT_REACTION_OFFSETS))
            .buf_arg(Args.points, GPU.memory.get_buffer(POINT))
            .buf_arg(Args.anti_gravity, GPU.memory.get_buffer(POINT_ANTI_GRAV))
            .buf_arg(Args.point_flags, GPU.memory.get_buffer(POINT_FLAG))
            .buf_arg(Args.point_hit_counts, GPU.memory.get_buffer(POINT_HIT_COUNT))
            .buf_arg(Args.point_hull_indices, GPU.memory.get_buffer(POINT_HULL_INDEX))
            .buf_arg(Args.hull_flags, GPU.memory.get_buffer(HULL_FLAG));
    }
}
