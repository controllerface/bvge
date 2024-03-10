package com.controllerface.bvge.cl.programs;

import com.controllerface.bvge.cl.CLUtils;
import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.cl.GPUProgram;

public class RootHullFilter extends GPUProgram
{
    @Override
    public void init()
    {
        src.add(CLUtils.read_src("programs/root_hull_filter.cl"));

        make_program();

        load_kernel(GPU.Kernel.root_hull_count);
        load_kernel(GPU.Kernel.root_hull_filter);
    }
}
