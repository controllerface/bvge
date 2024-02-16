package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.cl.GPUKernel;

public class PrepareBones_k extends GPUKernel
{
    private static final GPU.Program program = GPU.Program.prepare_bones;
    private static final GPU.Kernel kernel = GPU.Kernel.prepare_bones;

    public enum Args
    {
        bones,
        bone_references,
        bone_index,
        hulls,
        armatures,
        hull_flags,
        vbo,
        offset;
    }

    public PrepareBones_k(long command_queue_ptr)
    {
        super(command_queue_ptr, program.kernel_ptr(kernel));
    }
}
