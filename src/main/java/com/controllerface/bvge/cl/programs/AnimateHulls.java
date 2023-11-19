package com.controllerface.bvge.cl.programs;

import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.cl.GPUProgram;

import static com.controllerface.bvge.cl.CLUtils.read_src;

public class AnimateHulls extends GPUProgram
{
    @Override
    protected void init()
    {
        src.add(const_hull_flags);
        src.add(func_matrix_transform);
        src.add(read_src("kernels/animate_hulls.cl"));

        make_program();

        load_kernel(GPU.Kernel.animate_hulls);
    }
}
