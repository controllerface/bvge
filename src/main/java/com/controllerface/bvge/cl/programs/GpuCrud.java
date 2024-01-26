package com.controllerface.bvge.cl.programs;

import com.controllerface.bvge.cl.GPUProgram;

import static com.controllerface.bvge.cl.GPU.*;
import static com.controllerface.bvge.cl.CLUtils.read_src;

public class GpuCrud extends GPUProgram
{
    @Override
    protected void init()
    {
        src.add(func_rotate_point);
        src.add(read_src("programs/gpu_crud.cl"));

        make_program();

        load_kernel(Kernel.update_accel);
        load_kernel(Kernel.rotate_hull);
        load_kernel(Kernel.read_position);
        load_kernel(Kernel.create_hull);
        load_kernel(Kernel.create_point);
        load_kernel(Kernel.create_edge);
        load_kernel(Kernel.create_bone_reference);
        load_kernel(Kernel.create_texture_uv);
        load_kernel(Kernel.create_vertex_reference);
        load_kernel(Kernel.create_mesh_reference);
        load_kernel(Kernel.create_mesh_face);
        load_kernel(Kernel.create_bone);
        load_kernel(Kernel.create_armature);
        load_kernel(Kernel.create_armature_bone);
        load_kernel(Kernel.create_bone_bind_pose);
    }
}
