package com.controllerface.bvge.cl.programs;

import com.controllerface.bvge.cl.GPUProgram;

import static com.controllerface.bvge.cl.GPU.*;
import static com.controllerface.bvge.cl.CLUtils.read_src;

public class AabbCollide extends GPUProgram
{
    @Override
    protected void init()
    {
        add_src(func_do_bounds_intersect);
        add_src(func_calculate_key_index);
        add_src(read_src("kernels/aabb_collide.cl"));

        make_program();

        make_kernel(kn_aabb_collide);
    }
}
