package com.controllerface.bvge.cl.programs;

import com.controllerface.bvge.cl.GPUProgram;

import static com.controllerface.bvge.cl.GPU.*;
import static com.controllerface.bvge.cl.CLUtils.read_src;

public class PrepareTransforms extends GPUProgram
{
    @Override
    protected void init()
    {
        add_src(read_src("kernels/prepare_transforms.cl"));

        make_program();

        make_kernel(Kernel.prepare_transforms);
    }
}
