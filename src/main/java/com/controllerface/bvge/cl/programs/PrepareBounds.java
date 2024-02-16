package com.controllerface.bvge.cl.programs;

import com.controllerface.bvge.cl.GPUProgram;

import static com.controllerface.bvge.cl.CLUtils.read_src;
import static com.controllerface.bvge.cl.GPU.Kernel;

public class PrepareBounds extends GPUProgram
{
    @Override
    protected void init()
    {
        src.add(read_src("programs/prepare_bounds.cl"));

        make_program();

        load_kernel(Kernel.prepare_bounds);
    }
}
