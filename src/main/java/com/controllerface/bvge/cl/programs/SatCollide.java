package com.controllerface.bvge.cl.programs;

import com.controllerface.bvge.cl.GpuKernel;

import static com.controllerface.bvge.cl.OpenCL.*;
import static com.controllerface.bvge.cl.OpenCLUtils.read_src;

public class SatCollide extends GpuKernel
{
    @Override
    protected void init()
    {
        var source = read_src("kernels/sat_collide.cl");

        this.program = cl_p(func_angle_between,
            func_calculate_centroid,
            func_closest_point_circle,
            func_project_circle,
            func_project_polygon,
            func_polygon_distance,
            func_edge_contact,
            source);

        // example loading kernel
        this.kernels.put(kn_sat_collide, cl_k(program, kn_sat_collide));
    }
}
