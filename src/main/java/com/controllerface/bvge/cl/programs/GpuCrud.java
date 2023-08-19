package com.controllerface.bvge.cl.programs;

import com.controllerface.bvge.cl.GPUProgram;

import static com.controllerface.bvge.cl.GPU.*;
import static com.controllerface.bvge.cl.CLUtils.read_src;

public class GpuCrud extends GPUProgram
{
    @Override
    protected void init()
    {
        add_src(func_rotate_point);
        add_src(read_src("kernels/gpu_crud.cl"));

        make_program();

        make_kernel(Kernel.update_accel);
        make_kernel(Kernel.rotate_hull);
        make_kernel(Kernel.read_position);
        make_kernel(Kernel.create_hull);
        make_kernel(Kernel.create_point);
        make_kernel(Kernel.create_edge);
        make_kernel(Kernel.create_bone_reference);
        make_kernel(Kernel.create_vertex_reference);
        make_kernel(Kernel.create_bone);
        make_kernel(Kernel.create_armature);
    }
}
