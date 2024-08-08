package com.controllerface.bvge.gpu.cl.programs.physics;

import com.controllerface.bvge.gpu.GPU;
import com.controllerface.bvge.gpu.cl.kernels.KernelType;
import com.controllerface.bvge.gpu.cl.programs.GPUProgram;

public class SatCollide extends GPUProgram
{
    @Override
    public GPUProgram init()
    {
        src.add(const_hit_thresholds);
        src.add(const_entity_flags);
        src.add(const_hull_flags);
        src.add(const_point_flags);
        src.add(const_edge_flags);
        src.add(func_angle_between);
        src.add(func_calculate_centroid);
        src.add(func_closest_point_circle);
        src.add(func_point_polygon_containment);
        src.add(func_project_circle);
        src.add(func_project_polygon);
        src.add(func_polygon_distance);
        src.add(func_edge_contact);
        src.add(func_circle_collision);
        src.add(func_polygon_collision);
        src.add(func_polygon_circle_collision);
        src.add(func_sensor_collision);
        src.add(func_block_collision);
        src.add(GPU.CL.read_src("programs/sat_collide.cl"));

        make_program();

        load_kernel(KernelType.sat_collide);
        load_kernel(KernelType.sort_reactions);
        load_kernel(KernelType.apply_reactions);
        load_kernel(KernelType.move_entities);
        load_kernel(KernelType.move_hulls);

        return this;
    }
}
