package com.controllerface.bvge.gpu.cl.kernels.physics;

import com.controllerface.bvge.gpu.cl.buffers.BufferGroup;
import com.controllerface.bvge.gpu.cl.contexts.CL_CommandQueue;
import com.controllerface.bvge.gpu.cl.kernels.GPUKernel;
import com.controllerface.bvge.gpu.cl.kernels.KernelType;
import com.controllerface.bvge.gpu.cl.programs.GPUProgram;
import com.controllerface.bvge.memory.types.PhysicsBufferType;

import static com.controllerface.bvge.memory.types.PhysicsBufferType.*;

public class FinalizeCandidates_k extends GPUKernel
{
    public enum Args
    {
        input_candidates,
        match_offsets,
        matches,
        used,
        counter,
        final_candidates,
        max_index,
    }

    public FinalizeCandidates_k(CL_CommandQueue command_queue_ptr, GPUProgram program)
    {
        super(command_queue_ptr, program.get_kernel(KernelType.finalize_candidates));
    }

    public GPUKernel init(BufferGroup<PhysicsBufferType> match_buffers, BufferGroup<PhysicsBufferType> candidate_buffers)
    {
        return this.buf_arg(Args.used, match_buffers.buffer(MATCHES_USED))
            .buf_arg(Args.matches, match_buffers.buffer(MATCHES))
            .buf_arg(Args.match_offsets, candidate_buffers.buffer(CANDIDATE_OFFSETS))
            .buf_arg(Args.input_candidates, candidate_buffers.buffer(CANDIDATE_COUNTS))
            .buf_arg(Args.final_candidates, candidate_buffers.buffer(CANDIDATES));

    }
}
