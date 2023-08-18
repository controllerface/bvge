package com.controllerface.bvge.cl.programs;

import com.controllerface.bvge.cl.GpuKernel;

import static com.controllerface.bvge.cl.OpenCL.*;
import static com.controllerface.bvge.cl.OpenCLUtils.read_src;

public class AnimateHulls extends GpuKernel
{
    @Override
    protected void init()
    {
        add_src(func_matrix_transform);
        add_src(read_src("kernels/animate_hulls.cl"));
        make_program();
        make_kernel(kn_animate_hulls);
    }
}
