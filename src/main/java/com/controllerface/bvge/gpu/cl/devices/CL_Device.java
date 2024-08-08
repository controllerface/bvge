package com.controllerface.bvge.gpu.cl.devices;

import com.controllerface.bvge.gpu.GPUResource;

import static org.lwjgl.opencl.CL10.CL_SUCCESS;
import static org.lwjgl.opencl.CL12.clReleaseDevice;

public record CL_Device(long ptr, long platform) implements GPUResource
{
    @Override
    public void release()
    {
        int result = clReleaseDevice(ptr);
        if (result != CL_SUCCESS) throw new RuntimeException("Error: clReleaseDevice()" + result);
    }
}
