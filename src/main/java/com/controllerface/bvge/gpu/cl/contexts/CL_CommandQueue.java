package com.controllerface.bvge.gpu.cl.contexts;

import com.controllerface.bvge.gpu.GPUResource;

import static org.lwjgl.opencl.CL10.*;

public record CL_CommandQueue(long ptr) implements GPUResource
{
    @Override
    public void release()
    {
        int result = clReleaseCommandQueue(ptr);
        if (result != CL_SUCCESS) throw new RuntimeException("Error: clReleaseCommandQueue()" + result);
    }

    public void finish()
    {
        clFinish(ptr);
    }
}
