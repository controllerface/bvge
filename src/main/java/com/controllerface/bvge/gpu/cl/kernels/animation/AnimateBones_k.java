package com.controllerface.bvge.gpu.cl.kernels.animation;

import com.controllerface.bvge.gpu.cl.contexts.CL_CommandQueue;
import com.controllerface.bvge.gpu.cl.kernels.GPUKernel;
import com.controllerface.bvge.gpu.cl.kernels.KernelType;
import com.controllerface.bvge.gpu.cl.programs.GPUProgram;

public class AnimateBones_k extends GPUKernel
{
    public enum Args
    {
        bones,
        bone_references,
        armature_bones,
        hull_bind_pose_indicies,
        hull_inv_bind_pose_indicies,
        max_hull_bone,
    }

    public AnimateBones_k(CL_CommandQueue command_queue_ptr, GPUProgram program)
    {
        super(command_queue_ptr, program.get_kernel(KernelType.animate_bones));
    }
}
