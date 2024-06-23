package com.controllerface.bvge.cl.kernels;

public class CompactHullBones_k extends GPUKernel
{
    public enum Args
    {
        hull_bone_shift,
        hull_bones,
        hull_bind_pose_indices,
        hull_inv_bind_pose_indices,
    }

    public CompactHullBones_k(long command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr, kernel_ptr);
    }
}
