package com.controllerface.bvge.gpu.cl.kernels.physics;

import com.controllerface.bvge.gpu.GPU;
import com.controllerface.bvge.gpu.cl.contexts.CL_CommandQueue;
import com.controllerface.bvge.gpu.cl.kernels.GPUKernel;
import com.controllerface.bvge.gpu.cl.kernels.KernelType;
import com.controllerface.bvge.gpu.cl.programs.GPUProgram;

import static com.controllerface.bvge.memory.types.CoreBufferType.*;

public class CalculateHullAABB_k extends GPUKernel
{
    public enum Args
    {
        hulls,
        hull_scales,
        hull_point_tables,
        hull_rotations,
        points,
        bounds,
        bounds_index_data,
        bounds_bank_data,
        hull_flags,
        args,
        max_hull,
    }

    public CalculateHullAABB_k(CL_CommandQueue command_queue_ptr, GPUProgram program)
    {
        super(command_queue_ptr, program.get_kernel(KernelType.calculate_hull_aabb));
    }

    public GPUKernel init()
    {
        return this.buf_arg(Args.hulls, GPU.memory.get_buffer(HULL))
            .buf_arg(Args.hull_scales, GPU.memory.get_buffer(HULL_SCALE))
            .buf_arg(Args.hull_point_tables, GPU.memory.get_buffer(HULL_POINT_TABLE))
            .buf_arg(Args.hull_rotations, GPU.memory.get_buffer(HULL_ROTATION))
            .buf_arg(Args.points, GPU.memory.get_buffer(POINT))
            .buf_arg(Args.bounds, GPU.memory.get_buffer(HULL_AABB))
            .buf_arg(Args.bounds_index_data, GPU.memory.get_buffer(HULL_AABB_INDEX))
            .buf_arg(Args.bounds_bank_data, GPU.memory.get_buffer(HULL_AABB_KEY_TABLE))
            .buf_arg(Args.hull_flags, GPU.memory.get_buffer(HULL_FLAG));
    }
}
