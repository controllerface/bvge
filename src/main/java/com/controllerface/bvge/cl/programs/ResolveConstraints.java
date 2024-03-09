package com.controllerface.bvge.cl.programs;

import com.controllerface.bvge.cl.GPUProgram;

import static com.controllerface.bvge.cl.CLUtils.read_src;
import static com.controllerface.bvge.cl.GPU.Kernel;

public class ResolveConstraints extends GPUProgram
{
    @Override
    public void init()
    {
        src.add(read_src("programs/resolve_constraints.cl"));

        make_program();

        load_kernel(Kernel.resolve_constraints);
    }
}
