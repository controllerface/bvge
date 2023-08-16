package com.controllerface.bvge.cl;

import org.jocl.cl_kernel;
import org.jocl.cl_program;

import java.util.HashMap;
import java.util.Map;

public abstract class GpuKernel
{
    protected cl_program program;

    protected Map<String, cl_kernel> kernels = new HashMap<>();

    protected abstract void init();

    public Map<String, cl_kernel> kernels()
    {
        return kernels;
    }
}
