package com.controllerface.bvge.cl.programs;

import com.controllerface.bvge.cl.CLUtils;
import com.controllerface.bvge.cl.kernels.Kernel;

public class PreparePoints extends GPUProgram
{
    @Override
    public void init()
    {
        src.add(const_hull_flags);
        src.add(CLUtils.read_src("programs/prepare_points.cl"));

        make_program();

        load_kernel(Kernel.prepare_points);
    }
}
