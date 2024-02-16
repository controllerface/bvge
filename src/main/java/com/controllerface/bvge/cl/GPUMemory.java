package com.controllerface.bvge.cl;

import org.jocl.cl_mem;
import org.lwjgl.opencl.CL12;

public class GPUMemory
{
    private final cl_mem src;
    private boolean released = false;

    public GPUMemory(cl_mem src)
    {
        this.src = src;
    }

    public cl_mem memory()
    {
        return src;
    }

    public long pointer()
    {
        return src.getNativePointer();
    }

    public void release()
    {
        if (!released)
        {
            CL12.clReleaseMemObject(src.getNativePointer());
            released = true;
        }
    }
}
