package com.controllerface.bvge.cl.kernels;

public class CreateTextureUV_k extends GPUKernel
{
    public enum Args
    {
        texture_uvs,
        target,
        new_texture_uv;
    }

    public CreateTextureUV_k(long command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr, kernel_ptr);
    }
}
