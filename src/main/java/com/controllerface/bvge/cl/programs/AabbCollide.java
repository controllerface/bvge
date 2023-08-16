package com.controllerface.bvge.cl.programs;

import com.controllerface.bvge.cl.GpuKernel;

import static com.controllerface.bvge.cl.OpenCL.*;
import static com.controllerface.bvge.cl.OpenCLUtils.read_src;

public class AabbCollide extends GpuKernel
{
    @Override
    protected void init()
    {
        var source = read_src("kernels/aabb_collide.cl");
        this.program = cl_p(func_do_bounds_intersect, func_calculate_key_index, source);
        this.kernels.put(kn_aabb_collide, cl_k(program, kn_aabb_collide));
    }
}
