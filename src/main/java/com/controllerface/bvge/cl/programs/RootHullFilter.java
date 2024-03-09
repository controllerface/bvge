package com.controllerface.bvge.cl.programs;

import com.controllerface.bvge.cl.GPUProgram;

import static com.controllerface.bvge.cl.CLUtils.read_src;
import static com.controllerface.bvge.cl.GPU.Kernel;

public class RootHullFilter extends GPUProgram
{
    @Override
    public void init()
    {
        src.add(read_src("programs/root_hull_filter.cl"));

        make_program();

        load_kernel(Kernel.root_hull_count);
        load_kernel(Kernel.root_hull_filter);
    }
}
