package com.controllerface.bvge.cl.programs;

import com.controllerface.bvge.cl.GPUProgram;

import static com.controllerface.bvge.cl.GPU.*;
import static com.controllerface.bvge.cl.CLUtils.read_src;

public class ResolveConstraints extends GPUProgram
{
    @Override
    protected void init()
    {
        add_src(read_src("kernels/resolve_constraints.cl"));

        make_program();

        make_kernel(Kernel.resolve_constraints);
    }
}
