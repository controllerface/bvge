package com.controllerface.bvge.cl.programs;

import com.controllerface.bvge.cl.CLUtils;
import com.controllerface.bvge.cl.kernels.Kernel;

public class CcdCollide extends GPUProgram
{
    @Override
    public GPUProgram init()
    {
        src.add(const_hull_flags);
        src.add(func_do_bounds_intersect);
        src.add(func_calculate_key_index);
        src.add(func_vector_lerp);
        src.add(CLUtils.read_src("programs/ccd_collide.cl"));

        make_program();

        load_kernel(Kernel.ccd_collide);
        load_kernel(Kernel.ccd_react);

        return this;
    }
}
