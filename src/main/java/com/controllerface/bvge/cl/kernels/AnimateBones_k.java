package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.cl.GPUKernel;

public class AnimateBones_k extends GPUKernel
{
    private static final GPU.Program program = GPU.Program.animate_hulls;
    private static final GPU.Kernel kernel = GPU.Kernel.animate_bones;

    public enum Args
    {
        bones,
        bone_references,
        armature_bones,
        bone_index_tables;
    }

    public AnimateBones_k(long command_queue_ptr)
    {
        super(command_queue_ptr, program.kernel_ptr(kernel));
    }
}
