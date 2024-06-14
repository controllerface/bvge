package com.controllerface.bvge.cl.kernels;

public class CreateBoneBindPose_k extends GPUKernel
{
    public enum Args
    {
        bone_bind_poses,
        target,
        new_bone_bind_pose,
    }

    public CreateBoneBindPose_k(long command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr, kernel_ptr);
    }
}
