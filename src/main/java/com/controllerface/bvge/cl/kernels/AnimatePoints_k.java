package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPUKernel;

public class AnimatePoints_k extends GPUKernel
{
    public enum Args
    {
        points,
        hulls,
        hull_flags,
        vertex_tables,
        bone_tables,
        vertex_weights,
        armatures,
        vertex_references,
        bones;
    }

    public AnimatePoints_k(long command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr, kernel_ptr);
    }
}
