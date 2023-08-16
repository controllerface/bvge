package com.controllerface.bvge.cl.programs;

import com.controllerface.bvge.cl.GpuKernel;

import static com.controllerface.bvge.cl.OpenCL.*;
import static com.controllerface.bvge.cl.OpenCLUtils.read_src;

public class AnimateHulls extends GpuKernel
{
    @Override
    protected void init()
    {
        var source = read_src("kernels/animate_hulls.cl");
        this.program = cl_p(func_matrix_transform, source);
        this.kernels.put(kn_animate_hulls, cl_k(program, kn_animate_hulls));
    }
}
