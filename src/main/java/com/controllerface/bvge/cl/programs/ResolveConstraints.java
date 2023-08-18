package com.controllerface.bvge.cl.programs;

import com.controllerface.bvge.cl.GpuKernel;

import static com.controllerface.bvge.cl.OpenCL.*;
import static com.controllerface.bvge.cl.OpenCLUtils.read_src;

public class ResolveConstraints extends GpuKernel
{
    @Override
    protected void init()
    {
        add_src(read_src("kernels/resolve_constraints.cl"));
        make_program();
        make_kernel(kn_resolve_constraints);
    }
}
