package com.controllerface.bvge.gpu.cl.programs;

import com.controllerface.bvge.gpu.cl.CLUtils;
import com.controllerface.bvge.gpu.cl.kernels.Kernel;

public class ResolveConstraints extends GPUProgram
{
    @Override
    public GPUProgram init()
    {
        src.add(const_edge_flags);
        src.add(const_hull_flags);
        src.add(CLUtils.read_src("programs/resolve_constraints.cl"));

        make_program();

        load_kernel(Kernel.resolve_constraints);

        return this;
    }
}
