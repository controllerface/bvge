package com.controllerface.bvge.gpu.cl.programs;

import com.controllerface.bvge.gpu.cl.kernels.Kernel;
import com.controllerface.bvge.gpu.cl.CLUtils;

public class PrepareTransforms extends GPUProgram
{
    @Override
    public GPUProgram init()
    {
        src.add(CLUtils.read_src("programs/prepare_transforms.cl"));

        make_program();

        load_kernel(Kernel.prepare_transforms);

        return this;
    }
}
