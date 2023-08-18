package com.controllerface.bvge.cl;

import org.jocl.cl_kernel;
import org.jocl.cl_program;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static com.controllerface.bvge.cl.OpenCL.*;

public abstract class GpuKernel
{
    protected cl_program program;
    protected Map<String, cl_kernel> kernels = new HashMap<>();
    protected List<String> source_files = new ArrayList<>();

    protected abstract void init();

    public Map<String, cl_kernel> kernels()
    {
        return kernels;
    }

    protected void add_src(String src)
    {
        source_files.add(src);
    }

    protected void make_program()
    {
        this.program = cl_p(this.source_files);
    }

    protected void make_kernel(String kernel_name)
    {
        this.kernels.put(kernel_name, cl_k(program, kernel_name));
    }
}
