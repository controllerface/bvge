package com.controllerface.bvge.cl.kernels;

public class CalculatePointAABB_k extends GPUKernel
{
    public enum Args
    {
        points,
        point_aabb,
        point_aabb_index,
        point_aabb_key_table,
        args,
        max_point,
    }

    public CalculatePointAABB_k(long command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr, kernel_ptr);
    }
}
