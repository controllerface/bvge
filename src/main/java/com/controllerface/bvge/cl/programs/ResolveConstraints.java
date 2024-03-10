package com.controllerface.bvge.cl.programs;

import com.controllerface.bvge.cl.CLUtils;
import com.controllerface.bvge.cl.GPUProgram;
import com.controllerface.bvge.cl.Kernel;

public class ResolveConstraints extends GPUProgram
{
    @Override
    public void init()
    {
        src.add(CLUtils.read_src("programs/resolve_constraints.cl"));

        make_program();

        load_kernel(Kernel.resolve_constraints);
    }
}
