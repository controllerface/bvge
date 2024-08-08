package com.controllerface.bvge.gpu.cl.programs;

import com.controllerface.bvge.gpu.GPU;
import com.controllerface.bvge.gpu.cl.CLUtils;
import com.controllerface.bvge.gpu.cl.kernels.KernelType;

public class PrepareEntities extends GPUProgram
{
    @Override
    public GPUProgram init()
    {
        src.add(GPU.CL.read_src("programs/prepare_entities.cl"));

        make_program();

        load_kernel(KernelType.prepare_entities);

        return this;
    }
}
