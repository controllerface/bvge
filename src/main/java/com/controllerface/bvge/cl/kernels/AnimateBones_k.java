package com.controllerface.bvge.cl.kernels;

public class AnimateBones_k extends GPUKernel
{
    public enum Args
    {
        bones,
        bone_references,
        armature_bones,
        hull_bind_pose_indicies,
        hull_inv_bind_pose_indicies,
    }

    public AnimateBones_k(long command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr, kernel_ptr);
    }
}
