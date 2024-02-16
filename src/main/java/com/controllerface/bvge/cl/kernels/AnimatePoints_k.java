package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.cl.GPUKernel;

public class AnimatePoints_k extends GPUKernel
{
    private static final GPU.Program program = GPU.Program.animate_hulls;
    private static final GPU.Kernel kernel = GPU.Kernel.animate_points;

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

    public AnimatePoints_k(long command_queue_ptr)
    {
        super(command_queue_ptr, program.kernel_ptr(kernel));
    }
}
