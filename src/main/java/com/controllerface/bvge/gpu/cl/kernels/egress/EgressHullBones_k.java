package com.controllerface.bvge.gpu.cl.kernels.egress;

import com.controllerface.bvge.gpu.cl.buffers.ResizableBuffer;
import com.controllerface.bvge.gpu.cl.contexts.CL_CommandQueue;
import com.controllerface.bvge.gpu.cl.kernels.GPUKernel;
import com.controllerface.bvge.gpu.cl.kernels.KernelType;
import com.controllerface.bvge.gpu.cl.programs.GPUProgram;
import com.controllerface.bvge.memory.GPUCoreMemory;
import com.controllerface.bvge.memory.groups.UnorderedCoreBufferGroup;

import static com.controllerface.bvge.memory.types.CoreBufferType.*;

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

    public EgressHullBones_k(CL_CommandQueue command_queue_ptr, GPUProgram program)
    {
        super(command_queue_ptr, program.get_kernel(KernelType.egress_hull_bones));
    }

    public GPUKernel init(GPUCoreMemory core_memory,
                          UnorderedCoreBufferGroup sector_buffers,
                          ResizableBuffer b_hull_bone_shift)
    {
        return this.buf_arg(Args.hull_bones_in, core_memory.get_buffer(HULL_BONE))
            .buf_arg(Args.hull_bind_pose_indicies_in, core_memory.get_buffer(HULL_BONE_BIND_POSE))
            .buf_arg(Args.hull_inv_bind_pose_indicies_in, core_memory.get_buffer(HULL_BONE_INV_BIND_POSE))
            .buf_arg(Args.hull_bones_out, sector_buffers.buffer(HULL_BONE))
            .buf_arg(Args.hull_bind_pose_indicies_out, sector_buffers.buffer(HULL_BONE_BIND_POSE))
            .buf_arg(Args.hull_inv_bind_pose_indicies_out, sector_buffers.buffer(HULL_BONE_INV_BIND_POSE))
            .buf_arg(Args.new_hull_bones, b_hull_bone_shift);
    }
}
