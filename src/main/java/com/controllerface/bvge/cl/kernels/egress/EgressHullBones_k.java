package com.controllerface.bvge.cl.kernels.egress;

import com.controllerface.bvge.cl.kernels.GPUKernel;

public class EgressHullBones_k extends GPUKernel
{
    public enum Args
    {
        hull_bones_in,
        hull_bind_pose_indicies_in,
        hull_inv_bind_pose_indicies_in,
        hull_bones_out,
        hull_bind_pose_indicies_out,
        hull_inv_bind_pose_indicies_out,
        new_hull_bones,
        max_hull_bone,
    }

    public EgressHullBones_k(long command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr, kernel_ptr);
    }
}
