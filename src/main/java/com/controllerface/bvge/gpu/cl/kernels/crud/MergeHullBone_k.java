package com.controllerface.bvge.gpu.cl.kernels.crud;

import com.controllerface.bvge.gpu.cl.contexts.CL_CommandQueue;
import com.controllerface.bvge.gpu.cl.kernels.GPUKernel;
import com.controllerface.bvge.gpu.cl.kernels.KernelType;
import com.controllerface.bvge.gpu.cl.programs.GPUProgram;
import com.controllerface.bvge.memory.GPUCoreMemory;
import com.controllerface.bvge.memory.groups.CoreBufferGroup;

import static com.controllerface.bvge.memory.types.CoreBufferType.*;

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

    public GPUKernel init(GPUCoreMemory core_memory, CoreBufferGroup core_buffers)
    {
        return this.buf_arg(Args.hull_bones_in, core_buffers.buffer(HULL_BONE))
            .buf_arg(Args.hull_bind_pose_indicies_in, core_buffers.buffer(HULL_BONE_BIND_POSE))
            .buf_arg(Args.hull_inv_bind_pose_indicies_in, core_buffers.buffer(HULL_BONE_INV_BIND_POSE))
            .buf_arg(Args.hull_bones_out, core_memory.get_buffer(HULL_BONE))
            .buf_arg(Args.hull_bind_pose_indicies_out, core_memory.get_buffer(HULL_BONE_BIND_POSE))
            .buf_arg(Args.hull_inv_bind_pose_indicies_out, core_memory.get_buffer(HULL_BONE_INV_BIND_POSE));
    }
}
