package com.controllerface.bvge.gpu.cl.kernels.crud;

import com.controllerface.bvge.gpu.GPU;
import com.controllerface.bvge.gpu.cl.contexts.CL_CommandQueue;
import com.controllerface.bvge.gpu.cl.kernels.GPUKernel;
import com.controllerface.bvge.gpu.cl.kernels.KernelArg;
import com.controllerface.bvge.gpu.cl.kernels.KernelType;
import com.controllerface.bvge.gpu.cl.programs.GPUProgram;
import com.controllerface.bvge.memory.groups.ReferenceBufferGroup;

import static com.controllerface.bvge.gpu.cl.buffers.CL_DataTypes.cl_float16;
import static com.controllerface.bvge.gpu.cl.buffers.CL_DataTypes.cl_int;
import static com.controllerface.bvge.memory.types.ReferenceBufferType.BONE_BIND_POSE;
import static com.controllerface.bvge.memory.types.ReferenceBufferType.BONE_LAYER;

public class CreateBoneBindPose_k extends GPUKernel
{
    public static final String kernel_source = GPU.CL.crud_create_k_src(KernelType.create_bone_bind_pose, Args.class);

    public enum Args implements KernelArg
    {
        bone_bind_poses    (cl_float16.buffer_name()),
        bone_layers        (cl_int.buffer_name()),
        target             (cl_int.name()),
        new_bone_bind_pose (cl_float16.name()),
        new_bone_layer     (cl_int.name()),

        ;

        private final String cl_type;
        Args(String clType) { cl_type = clType; }
        public String cl_type() { return cl_type; }
    }

    public CreateBoneBindPose_k(CL_CommandQueue command_queue_ptr, GPUProgram program)
    {
        super(command_queue_ptr, program.get_kernel(KernelType.create_bone_bind_pose));
    }

    public GPUKernel init(ReferenceBufferGroup reference_buffers)
    {
        return this.buf_arg(Args.bone_bind_poses, reference_buffers.buffer(BONE_BIND_POSE))
            .buf_arg(Args.bone_layers, reference_buffers.buffer(BONE_LAYER));
    }
}
