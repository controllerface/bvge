package com.controllerface.bvge.gpu.cl.kernels.compact;

import com.controllerface.bvge.gpu.GPU;
import com.controllerface.bvge.gpu.cl.buffers.CL_DataTypes;
import com.controllerface.bvge.gpu.cl.buffers.ResizableBuffer;
import com.controllerface.bvge.gpu.cl.contexts.CL_CommandQueue;
import com.controllerface.bvge.gpu.cl.kernels.GPUKernel;
import com.controllerface.bvge.gpu.cl.kernels.KernelArg;
import com.controllerface.bvge.gpu.cl.kernels.KernelType;
import com.controllerface.bvge.gpu.cl.programs.GPUProgram;
import com.controllerface.bvge.memory.groups.CoreBufferGroup;

import static com.controllerface.bvge.memory.types.CoreBufferType.*;

public class CompactHullBones_k extends GPUKernel
{
    public static final String kernel_source = GPU.CL.compact_k_src(KernelType.compact_hull_bones, Args.class);

    public enum Args implements KernelArg
    {
        hull_bone_shift            (CL_DataTypes.cl_int.buffer_name()),
        hull_bones                 (HULL_BONE.data_type().buffer_name()),
        hull_bind_pose_indices     (HULL_BONE_BIND_POSE.data_type().buffer_name()),
        hull_inv_bind_pose_indices (HULL_BONE_INV_BIND_POSE.data_type().buffer_name()),

        ;

        private final String cl_type;
        Args(String clType) { cl_type = clType; }
        public String cl_type() { return cl_type; }
    }

    public CompactHullBones_k(CL_CommandQueue command_queue_ptr, GPUProgram program)
    {
        super(command_queue_ptr, program.get_kernel(KernelType.compact_hull_bones));
    }

    public GPUKernel init(CoreBufferGroup sector_buffers, ResizableBuffer b_hull_bone_shift)
    {
        return this.buf_arg(Args.hull_bone_shift, b_hull_bone_shift)
            .buf_arg(Args.hull_bones, sector_buffers.buffer(HULL_BONE))
            .buf_arg(Args.hull_bind_pose_indices, sector_buffers.buffer(HULL_BONE_BIND_POSE))
            .buf_arg(Args.hull_inv_bind_pose_indices, sector_buffers.buffer(HULL_BONE_INV_BIND_POSE));
    }
}
