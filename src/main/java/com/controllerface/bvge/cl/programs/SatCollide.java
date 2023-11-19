package com.controllerface.bvge.cl.programs;

import com.controllerface.bvge.cl.GPUProgram;

import static com.controllerface.bvge.cl.CLUtils.*;
import static com.controllerface.bvge.cl.GPU.*;

public class SatCollide extends GPUProgram
{
    @Override
    protected void init()
    {
        src.add(const_hull_flags);
        src.add(func_angle_between);
        src.add(func_calculate_centroid);
        src.add(func_closest_point_circle);
        src.add(func_project_circle);
        src.add(func_project_polygon);
        src.add(func_polygon_distance);
        src.add(func_edge_contact);
        src.add(func_circle_collision);
        src.add(func_polygon_collision);
        src.add(func_polygon_circle_collision);
        src.add(read_src("kernels/sat_collide.cl"));

        make_program();

        load_kernel(Kernel.sat_collide);
        load_kernel(Kernel.sort_reactions);
        load_kernel(Kernel.apply_reactions);
        load_kernel(Kernel.move_armatures);
    }
}
