package com.controllerface.bvge.gpu.cl.programs.rendering;

import com.controllerface.bvge.gpu.GPU;
import com.controllerface.bvge.gpu.cl.kernels.KernelType;
import com.controllerface.bvge.gpu.cl.programs.GPUProgram;
import com.controllerface.bvge.substances.Liquid;

public class PrepareLiquids extends GPUProgram
{
    @Override
    public GPUProgram init()
    {
        src.add(Liquid.cl_lookup_table());
        src.add(const_hit_thresholds);
        src.add(GPU.CL.read_src("programs/prepare_liquids.cl"));

        make_program();

        load_kernel(KernelType.prepare_liquids);

        return this;
    }
}
