package com.controllerface.bvge.gpu.cl.kernels.physics;

import com.controllerface.bvge.gpu.GPU;
import com.controllerface.bvge.gpu.cl.buffers.BufferGroup;
import com.controllerface.bvge.gpu.cl.buffers.CL_Buffer;
import com.controllerface.bvge.gpu.cl.contexts.CL_CommandQueue;
import com.controllerface.bvge.gpu.cl.kernels.GPUKernel;
import com.controllerface.bvge.gpu.cl.kernels.KernelType;
import com.controllerface.bvge.gpu.cl.programs.GPUProgram;
import com.controllerface.bvge.memory.types.PhysicsBufferType;
import com.controllerface.bvge.physics.UniformGrid;

import static com.controllerface.bvge.memory.types.CoreBufferType.HULL_AABB_KEY_TABLE;
import static com.controllerface.bvge.memory.types.PhysicsBufferType.*;

public class CountCandidates_k extends GPUKernel
{
    public enum Args
    {
        bounds_bank_data,
        in_bounds,
        key_bank,
        key_counts,
        candidates,
        x_subdivisions,
        key_count_length,
        max_index,
    }

    public CountCandidates_k(CL_CommandQueue command_queue_ptr, GPUProgram program)
    {
        super(command_queue_ptr, program.get_kernel(KernelType.count_candidates));
    }

    public GPUKernel init(BufferGroup<PhysicsBufferType> candidate_buffers,
                          BufferGroup<PhysicsBufferType> key_buffers,
                          CL_Buffer counts_buf,
                          UniformGrid uniform_grid)
    {
        return this.buf_arg(Args.candidates, candidate_buffers.buffer(CANDIDATE_COUNTS))
            .buf_arg(Args.key_bank, key_buffers.buffer(KEY_BANK))
            .buf_arg(Args.in_bounds, candidate_buffers.buffer(IN_BOUNDS))
            .buf_arg(Args.bounds_bank_data, GPU.memory.get_buffer(HULL_AABB_KEY_TABLE))
            .buf_arg(Args.key_counts, counts_buf)
            .set_arg(Args.x_subdivisions, uniform_grid.x_subdivisions)
            .set_arg(Args.key_count_length, uniform_grid.directory_length);
    }
}
