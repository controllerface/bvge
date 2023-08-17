package com.controllerface.bvge.cl.programs;

import com.controllerface.bvge.cl.GpuKernel;

import static com.controllerface.bvge.cl.OpenCL.*;
import static com.controllerface.bvge.cl.OpenCLUtils.read_src;

public class ClampPointVelocity extends GpuKernel
{
    @Override
    protected void init()
    {
        var source = read_src("kernels/clamp_point_velocity.cl");
        this.program = cl_p(source);
        this.kernels.put(kn_clamp_point_velocity, cl_k(program, kn_clamp_point_velocity));
    }
}
