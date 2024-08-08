package com.controllerface.bvge.gpu.cl.buffers;

import com.controllerface.bvge.gpu.GPUResource;

import static org.lwjgl.opencl.CL10.CL_SUCCESS;
import static org.lwjgl.opencl.CL10.clReleaseMemObject;

public record CL_Buffer(long ptr) implements GPUResource
{
    @Override
    public void release()
    {
        int result = clReleaseMemObject(ptr);
        if (result != CL_SUCCESS) throw new RuntimeException("Error: clReleaseMemObject()" + result);
    }
}
