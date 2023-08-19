package com.controllerface.bvge.cl.programs;

import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.cl.GPUProgram;

import static com.controllerface.bvge.cl.CLUtils.read_src;

public class AnimateHulls extends GPUProgram
{
    @Override
    protected void init()
    {
        add_src(func_matrix_transform);
        add_src(read_src("kernels/animate_hulls.cl"));

        make_program();

        make_kernel(GPU.Kernel.animate_hulls);
    }
}
