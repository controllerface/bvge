package com.controllerface.bvge.cl.kernels;

public class CreateBoneRef_k extends GPUKernel
{

    public enum Args
    {
        bone_references,
        target,
        new_bone_reference;
    }

    public CreateBoneRef_k(long command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr, kernel_ptr);
    }
}
