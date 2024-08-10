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

import static com.controllerface.bvge.memory.types.CoreBufferType.HULL_AABB_INDEX;
import static com.controllerface.bvge.memory.types.CoreBufferType.HULL_AABB_KEY_TABLE;
import static com.controllerface.bvge.memory.types.PhysicsBufferType.KEY_BANK;

public class BuildKeyBank_k extends GPUKernel
{
    public enum Args
    {
        hull_aabb_index,
        hull_aabb_key_table,
        key_bank,
        key_counts,
        x_subdivisions,
        key_bank_length,
        key_count_length,
        max_hull,
    }

    public BuildKeyBank_k(CL_CommandQueue command_queue_ptr, GPUProgram program)
    {
        super(command_queue_ptr, program.get_kernel(KernelType.build_key_bank));
    }

    public GPUKernel init(BufferGroup<PhysicsBufferType> key_buffers,
                          CL_Buffer counts_buf,
                          UniformGrid uniform_grid)
    {
        return this.buf_arg(Args.hull_aabb_index, GPU.memory.get_buffer(HULL_AABB_INDEX))
            .buf_arg(Args.hull_aabb_key_table, GPU.memory.get_buffer(HULL_AABB_KEY_TABLE))
            .buf_arg(Args.key_bank, key_buffers.buffer(KEY_BANK))
            .buf_arg(Args.key_counts, counts_buf)
            .set_arg(Args.x_subdivisions, uniform_grid.x_subdivisions)
            .set_arg(Args.key_count_length, uniform_grid.directory_length);
    }
}
