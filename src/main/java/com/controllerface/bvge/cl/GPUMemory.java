package com.controllerface.bvge.cl;

import static org.lwjgl.opencl.CL12.clReleaseMemObject;

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
            clReleaseMemObject(pointer);
            released = true;
        }
    }
}
