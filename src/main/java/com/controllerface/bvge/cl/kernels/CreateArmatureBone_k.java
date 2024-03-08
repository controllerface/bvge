package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.cl.GPUKernel;

public class CreateArmatureBone_k extends GPUKernel
{
    private static final GPU.Program program = GPU.Program.gpu_crud;
    private static final GPU.Kernel kernel = GPU.Kernel.create_armature_bone;

    public enum Args
    {
        armature_bones,
        bone_bind_tables,
        target,
        new_armature_bone,
        new_bone_bind_table;
    }

    public CreateArmatureBone_k(long command_queue_ptr)
    {
        super(command_queue_ptr, program.gpu.kernel_ptr(kernel));
    }
}
