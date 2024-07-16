package com.controllerface.bvge.cl.kernels.physics;

import com.controllerface.bvge.cl.kernels.GPUKernel;

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

    public CalculateHullAABB_k(long command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr, kernel_ptr);
    }
}
