package com.controllerface.bvge.cl.programs;

import com.controllerface.bvge.cl.GpuKernel;

import static com.controllerface.bvge.cl.OpenCL.*;
import static com.controllerface.bvge.cl.OpenCLUtils.read_src;

public class Integrate extends GpuKernel
{
    @Override
    protected void init()
    {
        var source = read_src("kernels/integrate.cl");

        this.program = cl_p(func_angle_between,
            func_rotate_point,
            func_is_in_bounds,
            func_get_extents,
            func_get_key_for_point,
            source);

        // example loading kernel
        this.kernels.put(kn_integrate, cl_k(program, kn_integrate));
    }
}
