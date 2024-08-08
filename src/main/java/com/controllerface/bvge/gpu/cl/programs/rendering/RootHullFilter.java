package com.controllerface.bvge.gpu.cl.programs.rendering;

import com.controllerface.bvge.gpu.GPU;
import com.controllerface.bvge.gpu.cl.kernels.KernelType;
import com.controllerface.bvge.gpu.cl.programs.GPUProgram;

public class RootHullFilter extends GPUProgram
{
    @Override
    public GPUProgram init()
    {
        src.add(GPU.CL.read_src("programs/root_hull_filter.cl"));

        make_program();

        load_kernel(KernelType.root_hull_count);
        load_kernel(KernelType.root_hull_filter);
        load_kernel(KernelType.hull_count);
        load_kernel(KernelType.hull_filter);

        return this;
    }
}
