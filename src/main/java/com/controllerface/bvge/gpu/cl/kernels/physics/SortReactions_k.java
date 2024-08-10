package com.controllerface.bvge.gpu.cl.kernels.physics;

import com.controllerface.bvge.gpu.cl.buffers.BufferGroup;
import com.controllerface.bvge.gpu.cl.contexts.CL_CommandQueue;
import com.controllerface.bvge.gpu.cl.kernels.GPUKernel;
import com.controllerface.bvge.gpu.cl.kernels.KernelType;
import com.controllerface.bvge.gpu.cl.programs.GPUProgram;
import com.controllerface.bvge.memory.types.PhysicsBufferType;

import static com.controllerface.bvge.memory.types.PhysicsBufferType.*;

public class SortReactions_k extends GPUKernel
{
    public enum Args
    {
        reactions_in,
        reactions_out,
        reaction_index,
        point_reactions,
        point_offsets,
        max_index,
    }

    public SortReactions_k(CL_CommandQueue command_queue_ptr, GPUProgram program)
    {
        super(command_queue_ptr, program.get_kernel(KernelType.sort_reactions));
    }

    public GPUKernel init(BufferGroup<PhysicsBufferType> reaction_buffers)
    {
        return this.buf_arg(Args.reactions_in, reaction_buffers.buffer(REACTIONS_IN))
            .buf_arg(Args.reactions_out, reaction_buffers.buffer(REACTIONS_OUT))
            .buf_arg(Args.reaction_index, reaction_buffers.buffer(REACTION_INDEX))
            .buf_arg(Args.point_reactions, reaction_buffers.buffer(POINT_REACTION_COUNTS))
            .buf_arg(Args.point_offsets, reaction_buffers.buffer(POINT_REACTION_OFFSETS));
    }
}
