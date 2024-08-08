package com.controllerface.bvge.gpu.cl.programs;

import com.controllerface.bvge.gpu.cl.kernels.Kernel;
import com.controllerface.bvge.gpu.cl.CLUtils;

public class RootHullFilter extends GPUProgram
{
    @Override
    public GPUProgram init()
    {
        src.add(CLUtils.read_src("programs/root_hull_filter.cl"));

        make_program();

        load_kernel(Kernel.root_hull_count);
        load_kernel(Kernel.root_hull_filter);
        load_kernel(Kernel.hull_count);
        load_kernel(Kernel.hull_filter);

        return this;
    }
}
