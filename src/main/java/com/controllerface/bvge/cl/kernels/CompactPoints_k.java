package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPUKernel;

public class CompactPoints_k extends GPUKernel
{
    public enum Args
    {
        point_shift,
        points,
        anti_gravity,
        vertex_tables,
        bone_tables;
    }

    public CompactPoints_k(long command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr, kernel_ptr);
    }
}
