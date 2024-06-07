package com.controllerface.bvge.cl.programs;

import com.controllerface.bvge.cl.CLUtils;
import com.controllerface.bvge.cl.kernels.Kernel;

public class PrepareLiquids extends GPUProgram
{
    @Override
    public void init()
    {
        src.add(const_liquid_lookup_table);
        src.add(const_hit_thresholds);
        src.add(CLUtils.read_src("programs/prepare_liquids.cl"));

        make_program();

        load_kernel(Kernel.prepare_liquids);
    }
}
