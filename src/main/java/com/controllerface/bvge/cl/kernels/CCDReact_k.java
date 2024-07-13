package com.controllerface.bvge.cl.kernels;

public class CCDReact_k extends GPUKernel
{
    public enum Args
    {
        points,
        point_anti_time,
        max_point,
    }

    public CCDReact_k(long command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr, kernel_ptr);
    }
}
