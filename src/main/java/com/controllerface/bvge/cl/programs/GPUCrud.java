package com.controllerface.bvge.cl.programs;

import com.controllerface.bvge.cl.CLUtils;
import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.cl.GPUProgram;

public class GPUCrud extends GPUProgram
{
    @Override
    public void init()
    {
        src.add(func_rotate_point);
        src.add(CLUtils.read_src("programs/gpu_crud.cl"));

        make_program();

        load_kernel(GPU.Kernel.create_animation_timings);
        load_kernel(GPU.Kernel.create_armature);
        load_kernel(GPU.Kernel.create_armature_bone);
        load_kernel(GPU.Kernel.create_bone);
        load_kernel(GPU.Kernel.create_bone_bind_pose);
        load_kernel(GPU.Kernel.create_bone_channel);
        load_kernel(GPU.Kernel.create_bone_reference);
        load_kernel(GPU.Kernel.create_edge);
        load_kernel(GPU.Kernel.create_hull);
        load_kernel(GPU.Kernel.create_keyframe);
        load_kernel(GPU.Kernel.create_mesh_face);
        load_kernel(GPU.Kernel.create_mesh_reference);
        load_kernel(GPU.Kernel.create_model_transform);
        load_kernel(GPU.Kernel.create_point);
        load_kernel(GPU.Kernel.create_texture_uv);
        load_kernel(GPU.Kernel.create_vertex_reference);
        load_kernel(GPU.Kernel.read_position);
        load_kernel(GPU.Kernel.rotate_hull);
        load_kernel(GPU.Kernel.set_bone_channel_table);
        load_kernel(GPU.Kernel.update_accel);
    }
}
