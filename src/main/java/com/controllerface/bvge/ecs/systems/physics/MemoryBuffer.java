package com.controllerface.bvge.ecs.systems.physics;

import com.controllerface.bvge.cl.OpenCL;
import org.jocl.Pointer;
import org.jocl.cl_mem;

import static org.jocl.CL.*;

public class MemoryBuffer
{
    private final cl_mem src;
    private final long size;
    private final Pointer dst;
    private boolean doTransfer = true;

    public MemoryBuffer(cl_mem src, long size, Pointer dst)
    {
        this.src = src;
        this.size = size;
        this.dst = dst;
    }

    public void setDoTransfer(boolean doTransfer)
    {
        this.doTransfer = doTransfer;
    }

    public cl_mem get_mem()
    {
        return src;
    }

    public long getSize()
    {
        return size;
    }

    public void transfer()
    {
        if (doTransfer)
        {
            clEnqueueReadBuffer(OpenCL.getCommandQueue(), src, CL_TRUE, 0, size, dst,
                0, null, null);
        }
        clReleaseMemObject(src);
    }
}
