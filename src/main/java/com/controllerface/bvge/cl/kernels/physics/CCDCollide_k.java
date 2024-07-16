package com.controllerface.bvge.cl.kernels.physics;

import com.controllerface.bvge.cl.kernels.GPUKernel;

public class CCDCollide_k extends GPUKernel
{
    public enum Args
    {
        edges,
        points,
        point_anti_time,
        edge_flags,
        candidates,
        max_index,
    }

    public CCDCollide_k(long command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr, kernel_ptr);
    }
}
