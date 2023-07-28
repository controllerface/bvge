package com.controllerface.bvge.ecs.systems.physics;

import com.controllerface.bvge.cl.OpenCL;
import org.jocl.Pointer;
import org.jocl.cl_command_queue;
import org.jocl.cl_mem;

import static org.jocl.CL.*;

public class MemoryBuffer
{
    private final cl_mem src;
    private final Pointer pointer;
    private boolean released = false;

    public MemoryBuffer(cl_mem src)
    {
        this.src = src;
        this.pointer = Pointer.to(this.src);
    }

    public cl_mem memory()
    {
        return src;
    }

    public Pointer pointer()
    {
        return pointer;
    }

    public void release()
    {
        if (!released)
        {
            clReleaseMemObject(src);
            released = true;
        }
    }
}
