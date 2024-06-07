package com.controllerface.bvge.cl.programs;

import com.controllerface.bvge.cl.CLUtils;
import com.controllerface.bvge.cl.kernels.Kernel;

public class Integrate extends GPUProgram
{
    @Override
    public void init()
    {
        src.add(const_hit_thresholds);
        src.add(const_point_flags);
        src.add(const_entity_flags);
        src.add(const_hull_flags);
        src.add(func_angle_between);
        src.add(func_rotate_point);
        src.add(func_is_in_bounds);
        src.add(func_get_extents);
        src.add(func_get_key_for_point);
        src.add(CLUtils.read_src("programs/integrate.cl"));

        make_program();

        load_kernel(Kernel.integrate);
        load_kernel(Kernel.integrate_entities);
    }
}
