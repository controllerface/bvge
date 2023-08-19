package com.controllerface.bvge.cl.programs;

import com.controllerface.bvge.cl.GPUProgram;

import static com.controllerface.bvge.cl.CLUtils.*;
import static com.controllerface.bvge.cl.GPU.*;

public class Integrate extends GPUProgram
{
    @Override
    protected void init()
    {
        add_src(func_angle_between);
        add_src(func_rotate_point);
        add_src(func_is_in_bounds);
        add_src(func_get_extents);
        add_src(func_get_key_for_point);
        add_src(read_src("kernels/integrate.cl"));

        make_program();

        make_kernel(Kernel.integrate);
    }
}
