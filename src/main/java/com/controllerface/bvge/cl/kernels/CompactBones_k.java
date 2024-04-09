package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPUKernel;

public class CompactBones_k extends GPUKernel
{
    public enum Args
    {
        bone_shift,
        bone_instances,
        hull_bind_pose_indicies,
        hull_inv_bind_pose_indicies,
    }

    public CompactBones_k(long command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr, kernel_ptr);
    }
}
