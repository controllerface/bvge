package com.controllerface.bvge.cl.programs;

import com.controllerface.bvge.cl.CLUtils;
import com.controllerface.bvge.cl.kernels.Kernel;

public class AabbCollide extends GPUProgram
{
    @Override
    public void init()
    {
        src.add(const_hull_flags);
        src.add(func_do_bounds_intersect);
        src.add(func_calculate_key_index);
        src.add(CLUtils.read_src("programs/aabb_collide.cl"));

        make_program();

        load_kernel(Kernel.aabb_collide);
    }
}
