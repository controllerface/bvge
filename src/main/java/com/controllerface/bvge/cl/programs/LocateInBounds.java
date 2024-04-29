package com.controllerface.bvge.cl.programs;

import com.controllerface.bvge.cl.CLUtils;
import com.controllerface.bvge.cl.GPUProgram;
import com.controllerface.bvge.cl.Kernel;

public class LocateInBounds extends GPUProgram
{
    @Override
    public void init()
    {
        src.add(prag_int32_base_atomics);
        src.add(const_hull_flags);
        src.add(func_do_bounds_intersect);
        src.add(func_calculate_key_index);
        src.add(CLUtils.read_src("programs/locate_in_bounds.cl"));

        make_program();

        load_kernel(Kernel.locate_in_bounds);
        load_kernel(Kernel.count_candidates);
        load_kernel(Kernel.finalize_candidates);
        load_kernel(Kernel.sat_sort_count);
        load_kernel(Kernel.sat_sort_type);
    }
}
