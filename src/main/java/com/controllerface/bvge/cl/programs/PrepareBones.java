package com.controllerface.bvge.cl.programs;

import com.controllerface.bvge.cl.GPUProgram;

import static com.controllerface.bvge.cl.GPU.*;
import static com.controllerface.bvge.cl.CLUtils.read_src;

public class PrepareBones extends GPUProgram
{
    @Override
    protected void init()
    {
        add_src(func_matrix_transform);
        add_src(read_src("kernels/prepare_bones.cl"));

        make_program();

        make_kernel(Kernel.prepare_bones);
    }
}
