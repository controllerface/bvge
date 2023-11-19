package com.controllerface.bvge.cl.programs;

import com.controllerface.bvge.cl.GPUProgram;

import static com.controllerface.bvge.cl.CLUtils.*;
import static com.controllerface.bvge.cl.GPU.*;

public class AabbCollide extends GPUProgram
{
    @Override
    protected void init()
    {
        src.add(const_hull_flags);
        src.add(func_do_bounds_intersect);
        src.add(func_calculate_key_index);
        src.add(read_src("kernels/aabb_collide.cl"));

        make_program();

        load_kernel(Kernel.aabb_collide);
    }
}
