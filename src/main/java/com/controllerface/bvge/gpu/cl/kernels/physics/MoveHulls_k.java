package com.controllerface.bvge.gpu.cl.kernels.physics;

import com.controllerface.bvge.gpu.cl.contexts.CL_CommandQueue;
import com.controllerface.bvge.gpu.cl.kernels.GPUKernel;

public class MoveHulls_k extends GPUKernel
{
    public enum Args
    {
        hulls,
        hull_point_tables,
        points,
        max_hull,
    }

    public MoveHulls_k(CL_CommandQueue command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr, kernel_ptr);
    }
}
