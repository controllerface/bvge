package com.controllerface.bvge.ecs.systems.physics;

import com.controllerface.bvge.cl.OpenCL;
import org.jocl.Pointer;
import org.jocl.cl_mem;

import static org.jocl.CL.*;

public class MemoryBuffer
{
    private final cl_mem src;
    private final Pointer pointer;
    private final long size;
    private final Pointer dst;
    private boolean doTransfer = true;
    private boolean releaseAfterTransfer = true;
    private boolean released = false;

    public MemoryBuffer(cl_mem src, long size, Pointer dst)
    {
        this.src = src;
        this.pointer = Pointer.to(this.src);
        this.size = size;
        this.dst = dst;
    }

    public void setCopyBuffer(boolean doCopy)
    {
        this.doTransfer = doCopy;
    }

    public void setReleaseAfterTransfer(boolean releaseAfterTransfer)
    {
        this.releaseAfterTransfer = releaseAfterTransfer;
    }

    public cl_mem memory()
    {
        return src;
    }

    public Pointer pointer()
    {
        return pointer;
    }

    public long getSize()
    {
        return size;
    }

    public void transfer()
    {
        if (!released && doTransfer)
        {
            clEnqueueReadBuffer(OpenCL.getCommandQueue(), src, CL_TRUE, 0, size, dst,
                0, null, null);
        }
        if (!released && releaseAfterTransfer)
        {
            release();
        }
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
