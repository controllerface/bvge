package com.controllerface.bvge.gpu.cl.contexts;

import com.controllerface.bvge.gpu.GPUResource;

import static org.lwjgl.opencl.CL10.CL_SUCCESS;
import static org.lwjgl.opencl.CL10.clReleaseContext;

public record CL_Context(long ptr) implements GPUResource
{
    @Override
    public void release()
    {
        int result = clReleaseContext(ptr);
        if (result != CL_SUCCESS) throw new RuntimeException("Error: clReleaseContext()" + result);
    }
}
