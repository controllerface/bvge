package com.controllerface.bvge.gpu.cl.programs;

import com.controllerface.bvge.gpu.GPUResource;

import static org.lwjgl.opencl.CL10.CL_SUCCESS;
import static org.lwjgl.opencl.CL10.clReleaseProgram;

public record CL_Program(long ptr) implements GPUResource
{
    @Override
    public void release()
    {
        int result = clReleaseProgram(ptr);
        if (result != CL_SUCCESS)
        {
            throw new RuntimeException("Error: clReleaseProgram()" + result);
        }
    }
}
