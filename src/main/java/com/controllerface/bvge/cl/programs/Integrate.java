package com.controllerface.bvge.cl.programs;

import com.controllerface.bvge.cl.GPUProgram;

import static com.controllerface.bvge.cl.CLUtils.*;
import static com.controllerface.bvge.cl.GPU.*;

public class Integrate extends GPUProgram
{
    @Override
    protected void init()
    {
        src.add(func_angle_between);
        src.add(func_rotate_point);
        src.add(func_is_in_bounds);
        src.add(func_get_extents);
        src.add(func_get_key_for_point);
        src.add(read_src("kernels/integrate.cl"));

        make_program();

        load_kernel(Kernel.integrate);
    }
}
