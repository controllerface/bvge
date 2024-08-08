package com.controllerface.bvge.gpu.cl.programs;

import com.controllerface.bvge.gpu.cl.CLUtils;
import com.controllerface.bvge.gpu.cl.kernels.Kernel;

public class LocateInBounds extends GPUProgram
{
    @Override
    public GPUProgram init()
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

        return this;
    }
}
