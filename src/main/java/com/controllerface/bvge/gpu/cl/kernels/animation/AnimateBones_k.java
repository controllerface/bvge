package com.controllerface.bvge.gpu.cl.kernels.animation;

import com.controllerface.bvge.gpu.GPU;
import com.controllerface.bvge.gpu.cl.contexts.CL_CommandQueue;
import com.controllerface.bvge.gpu.cl.kernels.GPUKernel;
import com.controllerface.bvge.gpu.cl.kernels.KernelType;
import com.controllerface.bvge.gpu.cl.programs.GPUProgram;

import static com.controllerface.bvge.memory.types.CoreBufferType.*;
import static com.controllerface.bvge.memory.types.ReferenceBufferType.BONE_REFERENCE;

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

    public GPUKernel init()
    {
        return this.buf_arg(Args.bones, GPU.memory.get_buffer(HULL_BONE))
            .buf_arg(Args.bone_references, GPU.memory.get_buffer(BONE_REFERENCE))
            .buf_arg(Args.armature_bones, GPU.memory.get_buffer(ENTITY_BONE))
            .buf_arg(Args.hull_bind_pose_indicies, GPU.memory.get_buffer(HULL_BONE_BIND_POSE))
            .buf_arg(Args.hull_inv_bind_pose_indicies, GPU.memory.get_buffer(HULL_BONE_INV_BIND_POSE));
    }
}
