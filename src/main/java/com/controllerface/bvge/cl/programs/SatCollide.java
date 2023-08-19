package com.controllerface.bvge.cl.programs;

import com.controllerface.bvge.cl.GPUProgram;

import static com.controllerface.bvge.cl.CLUtils.*;
import static com.controllerface.bvge.cl.GPU.*;

public class SatCollide extends GPUProgram
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

        make_kernel(Kernel.sat_collide);
    }
}
