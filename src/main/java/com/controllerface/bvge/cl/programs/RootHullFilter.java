package com.controllerface.bvge.cl.programs;

import com.controllerface.bvge.cl.CLUtils;
import com.controllerface.bvge.cl.kernels.Kernel;

public class RootHullFilter extends GPUProgram
{
    @Override
    public GPUProgram init()
    {
        src.add(CLUtils.read_src("programs/root_hull_filter.cl"));

        make_program();

        load_kernel(Kernel.root_hull_count);
        load_kernel(Kernel.root_hull_filter);

        return this;
    }
}
