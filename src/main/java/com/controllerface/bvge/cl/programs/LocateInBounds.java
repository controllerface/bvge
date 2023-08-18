package com.controllerface.bvge.cl.programs;

import com.controllerface.bvge.cl.GPUProgram;

import static com.controllerface.bvge.cl.GPU.*;
import static com.controllerface.bvge.cl.CLUtils.read_src;

public class LocateInBounds extends GPUProgram
{
    @Override
    protected void init()
    {
        add_src(prag_int32_base_atomics);
        add_src(func_do_bounds_intersect);
        add_src(func_calculate_key_index);
        add_src(read_src("kernels/locate_in_bounds.cl"));

        make_program();

        make_kernel(kn_locate_in_bounds);
        make_kernel(kn_count_candidates);
        make_kernel(kn_finalize_candidates);
    }
}
