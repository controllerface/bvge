package com.controllerface.bvge.cl;

import org.lwjgl.opencl.CL12;

public class GPUMemory
{
    private final long pointer;
    private boolean released = false;

    public GPUMemory(long pointer)
    {
        this.pointer = pointer;
    }

    public long pointer()
    {
        return pointer;
    }

    public void release()
    {
        if (!released)
        {
            CL12.clReleaseMemObject(pointer);
            released = true;
        }
    }
}
