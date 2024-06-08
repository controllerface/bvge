package com.controllerface.bvge.cl.programs;

import com.controllerface.bvge.cl.CLUtils;
import com.controllerface.bvge.cl.kernels.Kernel;

public class PrepareEntities extends GPUProgram
{
    @Override
    public GPUProgram init()
    {
        src.add(CLUtils.read_src("programs/prepare_entities.cl"));

        make_program();

        load_kernel(Kernel.prepare_entities);

        return this;
    }
}
