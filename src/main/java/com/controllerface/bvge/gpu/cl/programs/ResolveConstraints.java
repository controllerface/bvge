package com.controllerface.bvge.gpu.cl.programs;

import com.controllerface.bvge.gpu.GPU;
import com.controllerface.bvge.gpu.cl.CLUtils;
import com.controllerface.bvge.gpu.cl.kernels.KernelType;

public class ResolveConstraints extends GPUProgram
{
    @Override
    public GPUProgram init()
    {
        src.add(const_edge_flags);
        src.add(const_hull_flags);
        src.add(GPU.CL.read_src("programs/resolve_constraints.cl"));

        make_program();

        load_kernel(KernelType.resolve_constraints);

        return this;
    }
}
