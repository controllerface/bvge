package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.cl.GPUKernel;

public class CreateBoneBindPose_k extends GPUKernel
{
    private static final GPU.Program program = GPU.Program.gpu_crud;
    private static final GPU.Kernel kernel = GPU.Kernel.create_bone_bind_pose;

    public enum Args
    {
        bone_bind_poses,
        bone_bind_parents,
        target,
        new_bone_bind_pose,
        bone_bind_parent;
    }

    public CreateBoneBindPose_k(long command_queue_ptr)
    {
        super(command_queue_ptr, program.gpu.kernel_ptr(kernel));
    }
}
