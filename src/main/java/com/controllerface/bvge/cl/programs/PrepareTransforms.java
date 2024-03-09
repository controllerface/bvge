package com.controllerface.bvge.cl.programs;

import com.controllerface.bvge.cl.GPUProgram;

import static com.controllerface.bvge.cl.CLUtils.read_src;
import static com.controllerface.bvge.cl.GPU.Kernel;

public class PrepareTransforms extends GPUProgram
{
    @Override
    public void init()
    {
        src.add(read_src("programs/prepare_transforms.cl"));

        make_program();

        load_kernel(Kernel.prepare_transforms);
    }
}
