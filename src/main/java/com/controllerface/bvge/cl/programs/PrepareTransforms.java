package com.controllerface.bvge.cl.programs;

import com.controllerface.bvge.cl.GpuKernel;

import static com.controllerface.bvge.cl.OpenCL.*;
import static com.controllerface.bvge.cl.OpenCLUtils.read_src;

public class PrepareTransforms extends GpuKernel
{
    @Override
    protected void init()
    {
        add_src(read_src("kernels/prepare_transforms.cl"));
        make_program();
        make_kernel(kn_prepare_transforms);
    }
}
