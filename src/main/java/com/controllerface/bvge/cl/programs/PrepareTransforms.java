package com.controllerface.bvge.cl.programs;

import com.controllerface.bvge.cl.GpuKernel;

import static com.controllerface.bvge.cl.OpenCL.*;
import static com.controllerface.bvge.cl.OpenCLUtils.read_src;

public class PrepareTransforms extends GpuKernel
{
    @Override
    protected void init()
    {
        var source = read_src("kernels/prepare_transforms.cl");
        this.program = cl_p(source);
        this.kernels.put(kn_prepare_transforms, cl_k(program, kn_prepare_transforms));
    }
}
