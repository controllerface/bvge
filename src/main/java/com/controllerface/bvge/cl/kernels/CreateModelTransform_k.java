package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.cl.GPUKernel;

public class CreateModelTransform_k extends GPUKernel
{
    private static final GPU.Program program = GPU.Program.gpu_crud;
    private static final GPU.Kernel kernel = GPU.Kernel.create_model_transform;

    public enum Args
    {
        model_transforms,
        target,
        new_model_transform;
    }

    public CreateModelTransform_k(long command_queue_ptr)
    {
        super(command_queue_ptr, program.kernel_ptr(kernel));
    }
}
