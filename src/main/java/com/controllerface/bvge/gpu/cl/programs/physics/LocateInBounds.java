package com.controllerface.bvge.gpu.cl.programs.physics;

import com.controllerface.bvge.gpu.GPU;
import com.controllerface.bvge.gpu.cl.kernels.KernelType;
import com.controllerface.bvge.gpu.cl.programs.GPUProgram;

public class LocateInBounds extends GPUProgram
{
    @Override
    public GPUProgram init()
    {
        src.add(prag_int32_base_atomics);
        src.add(const_hull_flags);
        src.add(func_do_bounds_intersect);
        src.add(func_calculate_key_index);
        src.add(GPU.CL.read_src("programs/locate_in_bounds.cl"));

        make_program();

        load_kernel(KernelType.locate_in_bounds);
        load_kernel(KernelType.count_candidates);
        load_kernel(KernelType.finalize_candidates);

        return this;
    }
}
