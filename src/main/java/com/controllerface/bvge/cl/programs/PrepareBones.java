package com.controllerface.bvge.cl.programs;

import com.controllerface.bvge.cl.GPUProgram;

import static com.controllerface.bvge.cl.GPU.*;
import static com.controllerface.bvge.cl.CLUtils.read_src;

public class PrepareBones extends GPUProgram
{
    @Override
    protected void init()
    {
        src.add(func_matrix_transform);
        src.add(read_src("kernels/prepare_bones.cl"));

        make_program();

        load_kernel(Kernel.prepare_bones);
    }
}
