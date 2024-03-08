package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.cl.GPUKernel;

public class MoveArmatures_k extends GPUKernel
{
    private static final GPU.Program program = GPU.Program.sat_collide;
    private static final GPU.Kernel kernel = GPU.Kernel.move_armatures;

    public enum Args
    {
        hulls,
        armatures,
        hull_tables,
        element_tables,
        hull_flags,
        points;
    }

    public MoveArmatures_k(long command_queue_ptr)
    {
        super(command_queue_ptr, program.gpu.kernel_ptr(kernel));
    }
}
