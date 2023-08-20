package com.controllerface.bvge.cl.programs;

import com.controllerface.bvge.cl.GPUProgram;

import static com.controllerface.bvge.cl.GPU.*;
import static com.controllerface.bvge.cl.CLUtils.read_src;

public class PrepareBounds extends GPUProgram
{
    @Override
    protected void init()
    {
        src.add(read_src("kernels/prepare_bounds.cl"));

        make_program();

        load_kernel(Kernel.prepare_bounds);
    }
}
