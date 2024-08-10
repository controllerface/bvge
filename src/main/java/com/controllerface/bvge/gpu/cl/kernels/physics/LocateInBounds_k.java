package com.controllerface.bvge.gpu.cl.kernels.physics;

import com.controllerface.bvge.gpu.GPU;
import com.controllerface.bvge.gpu.cl.buffers.BufferGroup;
import com.controllerface.bvge.gpu.cl.contexts.CL_CommandQueue;
import com.controllerface.bvge.gpu.cl.kernels.GPUKernel;
import com.controllerface.bvge.gpu.cl.kernels.KernelType;
import com.controllerface.bvge.gpu.cl.programs.GPUProgram;
import com.controllerface.bvge.memory.types.PhysicsBufferType;

import static com.controllerface.bvge.memory.types.CoreBufferType.HULL_AABB_KEY_TABLE;
import static com.controllerface.bvge.memory.types.PhysicsBufferType.IN_BOUNDS;

public class LocateInBounds_k extends GPUKernel
{
    public enum Args
    {
        bounds_bank_data,
        in_bounds,
        counter,
        max_bound,
    }

    public LocateInBounds_k(CL_CommandQueue command_queue_ptr, GPUProgram program)
    {
        super(command_queue_ptr, program.get_kernel(KernelType.locate_in_bounds));
    }

    public GPUKernel init(BufferGroup<PhysicsBufferType> candidate_buffers)
    {
        return this.buf_arg(Args.in_bounds, candidate_buffers.buffer(IN_BOUNDS))
            .buf_arg(Args.bounds_bank_data, GPU.memory.get_buffer(HULL_AABB_KEY_TABLE));
    }
}
