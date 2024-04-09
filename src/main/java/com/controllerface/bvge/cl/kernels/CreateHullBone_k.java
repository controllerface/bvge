package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPUKernel;

public class CreateHullBone_k extends GPUKernel
{
    public enum Args
    {
        bones,
        hull_bind_pose_indicies,
        hull_inv_bind_pose_indicies,
        target,
        new_bone,
        new_hull_bind_pose_id,
        new_hull_inv_bind_pose_id,
    }

    public CreateHullBone_k(long command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr, kernel_ptr);
    }
}
