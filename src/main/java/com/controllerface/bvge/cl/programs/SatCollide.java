package com.controllerface.bvge.cl.programs;

import com.controllerface.bvge.cl.GpuKernel;

import static com.controllerface.bvge.cl.OpenCL.*;
import static com.controllerface.bvge.cl.OpenCLUtils.read_src;

public class SatCollide extends GpuKernel
{
    @Override
    protected void init()
    {
        add_src(func_angle_between);
        add_src(func_calculate_centroid);
        add_src(func_closest_point_circle);
        add_src(func_project_circle);
        add_src(func_project_polygon);
        add_src(func_polygon_distance);
        add_src(func_edge_contact);
        add_src(func_circle_collision);
        add_src(func_polygon_collision);
        add_src(func_polygon_circle_collision);
        add_src(read_src("kernels/sat_collide.cl"));
        make_program();
        make_kernel(kn_sat_collide);
    }
}
