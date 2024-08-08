package com.controllerface.bvge.gpu.cl.kernels.crud;

import com.controllerface.bvge.gpu.cl.contexts.CL_CommandQueue;
import com.controllerface.bvge.gpu.cl.kernels.GPUKernel;
import com.controllerface.bvge.gpu.cl.kernels.KernelType;
import com.controllerface.bvge.gpu.cl.programs.GPUProgram;

public class MergeHullBone_k extends GPUKernel
{
    public enum Args
    {
        hull_bones_in,
        hull_bind_pose_indicies_in,
        hull_inv_bind_pose_indicies_in,
        hull_bones_out,
        hull_bind_pose_indicies_out,
        hull_inv_bind_pose_indicies_out,
        hull_bone_offset,
        armature_bone_offset,
        max_hull_bone,
    }

    public MergeHullBone_k(CL_CommandQueue command_queue_ptr, GPUProgram program)
    {
        super(command_queue_ptr, program.get_kernel(KernelType.merge_hull_bone));
    }
}
