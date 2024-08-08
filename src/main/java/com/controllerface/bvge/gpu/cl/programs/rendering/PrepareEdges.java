package com.controllerface.bvge.gpu.cl.programs.rendering;

import com.controllerface.bvge.gpu.GPU;
import com.controllerface.bvge.gpu.cl.kernels.KernelType;
import com.controllerface.bvge.gpu.cl.programs.GPUProgram;

public class PrepareEdges extends GPUProgram
{
    @Override
    public GPUProgram init()
    {
        src.add(const_edge_flags);
        src.add(const_hull_flags);
        src.add(GPU.CL.read_src("programs/prepare_edges.cl"));

        make_program();

        load_kernel(KernelType.prepare_edges);

        return this;
    }
}
