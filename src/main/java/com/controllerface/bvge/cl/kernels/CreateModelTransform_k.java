package com.controllerface.bvge.cl.kernels;

public class CreateModelTransform_k extends GPUKernel
{
    public enum Args
    {
        model_transforms,
        target,
        new_model_transform;
    }

    public CreateModelTransform_k(long command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr, kernel_ptr);
    }
}
