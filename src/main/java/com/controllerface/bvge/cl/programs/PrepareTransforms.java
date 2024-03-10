package com.controllerface.bvge.cl.programs;

import com.controllerface.bvge.cl.CLUtils;
import com.controllerface.bvge.cl.GPUProgram;
import com.controllerface.bvge.cl.Kernel;

public class PrepareTransforms extends GPUProgram
{
    @Override
    public void init()
    {
        src.add(CLUtils.read_src("programs/prepare_transforms.cl"));

        make_program();

        load_kernel(Kernel.prepare_transforms);
    }
}
