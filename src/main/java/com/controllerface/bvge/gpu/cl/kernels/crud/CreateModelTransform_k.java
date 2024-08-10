package com.controllerface.bvge.gpu.cl.kernels.crud;

import com.controllerface.bvge.gpu.GPU;
import com.controllerface.bvge.gpu.cl.buffers.CL_DataTypes;
import com.controllerface.bvge.gpu.cl.contexts.CL_CommandQueue;
import com.controllerface.bvge.gpu.cl.kernels.GPUKernel;
import com.controllerface.bvge.gpu.cl.kernels.KernelArg;
import com.controllerface.bvge.gpu.cl.kernels.KernelType;
import com.controllerface.bvge.gpu.cl.programs.GPUProgram;
import com.controllerface.bvge.memory.groups.ReferenceBufferGroup;

import static com.controllerface.bvge.memory.types.ReferenceBufferType.MODEL_TRANSFORM;

public class CreateModelTransform_k extends GPUKernel
{
    public static final String kernel_source = GPU.CL.crud_create_k_src(KernelType.create_model_transform, Args.class);

    public enum Args implements KernelArg
    {
        model_transforms    (CL_DataTypes.cl_float16.buffer_name()),
        target              (CL_DataTypes.cl_int.name()),
        new_model_transform (CL_DataTypes.cl_float16.name()),

        ;

        private final String cl_type;
        Args(String clType) { cl_type = clType; }
        public String cl_type() { return cl_type; }
    }

    public CreateModelTransform_k(CL_CommandQueue command_queue_ptr, GPUProgram program)
    {
        super(command_queue_ptr, program.get_kernel(KernelType.create_model_transform));
    }

    public GPUKernel init(ReferenceBufferGroup reference_buffers)
    {
        return this.buf_arg(Args.model_transforms, reference_buffers.buffer(MODEL_TRANSFORM));
    }
}
