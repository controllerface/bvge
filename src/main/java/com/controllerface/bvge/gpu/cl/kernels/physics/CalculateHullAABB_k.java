package com.controllerface.bvge.gpu.cl.kernels.physics;

import com.controllerface.bvge.gpu.cl.contexts.CL_CommandQueue;
import com.controllerface.bvge.gpu.cl.kernels.GPUKernel;
import com.controllerface.bvge.gpu.cl.kernels.KernelType;
import com.controllerface.bvge.gpu.cl.programs.GPUProgram;

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
}
