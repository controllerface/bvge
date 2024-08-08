package com.controllerface.bvge.gpu.cl.programs;

import com.controllerface.bvge.gpu.GPU;
import com.controllerface.bvge.gpu.cl.CLUtils;
import com.controllerface.bvge.gpu.cl.kernels.KernelType;

public class Integrate extends GPUProgram
{
    @Override
    public GPUProgram init()
    {
        src.add(const_hit_thresholds);
        src.add(const_point_flags);
        src.add(const_entity_flags);
        src.add(const_hull_flags);
        src.add(func_angle_between);
        src.add(func_rotate_point);
        src.add(func_is_in_bounds);
        src.add(func_get_extents);
        src.add(func_get_key_for_point);
        src.add(GPU.CL.read_src("programs/integrate.cl"));

        make_program();

        load_kernel(KernelType.integrate);
        load_kernel(KernelType.integrate_entities);
        load_kernel(KernelType.calculate_hull_aabb);

        return this;
    }
}
