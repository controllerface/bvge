package com.controllerface.bvge.gpu.cl.kernels.crud;

import com.controllerface.bvge.gpu.GPU;
import com.controllerface.bvge.gpu.cl.buffers.CL_DataTypes;
import com.controllerface.bvge.gpu.cl.contexts.CL_CommandQueue;
import com.controllerface.bvge.gpu.cl.kernels.GPUKernel;
import com.controllerface.bvge.gpu.cl.kernels.KernelArg;
import com.controllerface.bvge.gpu.cl.kernels.KernelType;
import com.controllerface.bvge.gpu.cl.programs.GPUProgram;
import com.controllerface.bvge.memory.groups.CoreBufferGroup;

import static com.controllerface.bvge.memory.types.CoreBufferType.*;

public class CreateHullBone_k extends GPUKernel
{
    public static final String kernel_source = GPU.CL.crud_create_k_src(KernelType.create_hull_bone, Args.class);

    public enum Args implements KernelArg
    {
        hull_bones                  (CL_DataTypes.cl_float16.buffer_name()),
        hull_bind_pose_indicies     (CL_DataTypes.cl_int.buffer_name()),
        hull_inv_bind_pose_indicies (CL_DataTypes.cl_int.buffer_name()),
        target                      (CL_DataTypes.cl_int.name()),
        new_hull_bone               (CL_DataTypes.cl_float16.name()),
        new_hull_bind_pose_id       (CL_DataTypes.cl_int.name()),
        new_hull_inv_bind_pose_id   (CL_DataTypes.cl_int.name()),

        ;

        private final String cl_type;
        Args(String clType) { cl_type = clType; }
        public String cl_type() { return cl_type; }
    }

    public CreateHullBone_k(CL_CommandQueue command_queue_ptr, GPUProgram program)
    {
        super(command_queue_ptr, program.get_kernel(KernelType.create_hull_bone));
    }

    public GPUKernel init(CoreBufferGroup core_buffers)
    {
        return this.buf_arg(Args.hull_bones, core_buffers.buffer(HULL_BONE))
            .buf_arg(Args.hull_bind_pose_indicies, core_buffers.buffer(HULL_BONE_BIND_POSE))
            .buf_arg(Args.hull_inv_bind_pose_indicies, core_buffers.buffer(HULL_BONE_INV_BIND_POSE));
    }
}
