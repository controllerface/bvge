package com.controllerface.bvge.cl.programs;

import com.controllerface.bvge.cl.GpuKernel;

import static com.controllerface.bvge.cl.OpenCL.*;
import static com.controllerface.bvge.cl.OpenCLUtils.read_src;

public class LocateInBounds extends GpuKernel
{
    @Override
    protected void init()
    {
        var source = read_src("kernels/locate_in_bounds.cl");
        this.program = cl_p(prag_int32_base_atomics, func_do_bounds_intersect, func_calculate_key_index, source);
        this.kernels.put(kn_locate_in_bounds,    cl_k(program, kn_locate_in_bounds));
        this.kernels.put(kn_count_candidates,    cl_k(program, kn_count_candidates));
        this.kernels.put(kn_finalize_candidates, cl_k(program, kn_finalize_candidates));
    }
}
