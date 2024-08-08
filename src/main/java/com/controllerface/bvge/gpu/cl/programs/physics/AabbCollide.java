package com.controllerface.bvge.gpu.cl.programs.physics;

import com.controllerface.bvge.gpu.GPU;
import com.controllerface.bvge.gpu.cl.kernels.KernelType;
import com.controllerface.bvge.gpu.cl.programs.GPUProgram;

public class AabbCollide extends GPUProgram
{
    @Override
    public GPUProgram init()
    {
        src.add(const_hull_flags);
        src.add(func_do_bounds_intersect);
        src.add(func_calculate_key_index);
        src.add(GPU.CL.read_src("programs/aabb_collide.cl"));

        make_program();

        load_kernel(KernelType.aabb_collide);

        return this;
    }
}
