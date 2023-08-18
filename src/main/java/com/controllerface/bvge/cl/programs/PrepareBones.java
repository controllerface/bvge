package com.controllerface.bvge.cl.programs;

import com.controllerface.bvge.cl.GpuKernel;

import static com.controllerface.bvge.cl.OpenCL.*;
import static com.controllerface.bvge.cl.OpenCLUtils.read_src;

public class PrepareBones extends GpuKernel
{
    @Override
    protected void init()
    {
        add_src(func_matrix_transform);
        add_src(read_src("kernels/prepare_bones.cl"));
        make_program();
        make_kernel(kn_prepare_bones);
    }
}
