package com.controllerface.bvge.cl.programs;

import com.controllerface.bvge.cl.CLUtils;
import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.cl.GPUProgram;

public class PrepareBounds extends GPUProgram
{
    @Override
    public void init()
    {
        src.add(CLUtils.read_src("programs/prepare_bounds.cl"));

        make_program();

        load_kernel(GPU.Kernel.prepare_bounds);
    }
}
