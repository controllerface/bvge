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

import static com.controllerface.bvge.memory.types.CoreBufferType.*;
import static com.controllerface.bvge.memory.types.PhysicsBufferType.*;

public class AABBCollide_k extends GPUKernel
{
    public enum Args
    {
        bounds,
        bounds_bank_data,
        hull_entity_ids,
        hull_flags,
        candidates,
        match_offsets,
        key_map,
        key_bank,
        key_counts,
        key_offsets,
        matches,
        used,
        counter,
        x_subdivisions,
        key_count_length,
        max_index,
    }

    public AABBCollide_k(CL_CommandQueue command_queue_ptr, GPUProgram program)
    {
        super(command_queue_ptr, program.get_kernel(KernelType.aabb_collide));
    }

    public GPUKernel init(BufferGroup<PhysicsBufferType> match_buffers,
                          BufferGroup<PhysicsBufferType> candidate_buffers,
                          BufferGroup<PhysicsBufferType> key_buffers,
                          CL_Buffer ptr_counts_data,
                          CL_Buffer ptr_offsets_data,
                          CL_Buffer svm_atomic_counter,
                          UniformGrid uniform_grid)
    {
        return this.buf_arg(Args.used, match_buffers.buffer(MATCHES_USED))
            .buf_arg(Args.matches, match_buffers.buffer(MATCHES))
            .buf_arg(Args.match_offsets, candidate_buffers.buffer(CANDIDATE_OFFSETS))
            .buf_arg(Args.candidates, candidate_buffers.buffer(CANDIDATE_COUNTS))
            .buf_arg(Args.key_map, key_buffers.buffer(KEY_MAP))
            .buf_arg(Args.key_bank, key_buffers.buffer(KEY_BANK))
            .buf_arg(Args.bounds, GPU.memory.get_buffer(HULL_AABB))
            .buf_arg(Args.bounds_bank_data, GPU.memory.get_buffer(HULL_AABB_KEY_TABLE))
            .buf_arg(Args.hull_entity_ids, GPU.memory.get_buffer(HULL_ENTITY_ID))
            .buf_arg(Args.hull_flags, GPU.memory.get_buffer(HULL_FLAG))
            .buf_arg(Args.key_counts, ptr_counts_data)
            .buf_arg(Args.key_offsets, ptr_offsets_data)
            .buf_arg(Args.counter, svm_atomic_counter)
            .set_arg(Args.x_subdivisions, uniform_grid.x_subdivisions)
            .set_arg(Args.key_count_length, uniform_grid.directory_length);
    }
}
