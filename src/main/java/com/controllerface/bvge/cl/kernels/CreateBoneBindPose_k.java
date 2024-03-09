package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPUKernel;

public class CreateBoneBindPose_k extends GPUKernel
{
    public enum Args
    {
        bone_bind_poses,
        bone_bind_parents,
        target,
        new_bone_bind_pose,
        bone_bind_parent;
    }

    public CreateBoneBindPose_k(long command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr, kernel_ptr);
    }
}
