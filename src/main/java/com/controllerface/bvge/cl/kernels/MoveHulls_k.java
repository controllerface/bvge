package com.controllerface.bvge.cl.kernels;

public class MoveHulls_k extends GPUKernel
{
    public enum Args
    {
        hulls,
        hull_point_tables,
        points,
        max_hull,
    }

    public MoveHulls_k(long command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr, kernel_ptr);
    }
}
