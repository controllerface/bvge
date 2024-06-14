package com.controllerface.bvge.cl.programs;

import com.controllerface.bvge.cl.CLUtils;
import com.controllerface.bvge.cl.kernels.Kernel;

public class PrepareBounds extends GPUProgram
{
    @Override
    public GPUProgram init()
    {
        src.add(CLUtils.read_src("programs/prepare_bounds.cl"));

        make_program();

        load_kernel(Kernel.prepare_bounds);

        return this;
    }
}
