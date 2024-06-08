package com.controllerface.bvge.cl.programs;

import com.controllerface.bvge.cl.CLUtils;
import com.controllerface.bvge.cl.kernels.Kernel;

public class GPUCrud extends GPUProgram
{
    @Override
    public GPUProgram init()
    {
        src.add(const_entity_flags);
        src.add(func_rotate_point);
        src.add(CLUtils.read_src("programs/gpu_crud.cl"));

        make_program();

        load_kernel(Kernel.create_animation_timings);
        load_kernel(Kernel.create_entity);
        load_kernel(Kernel.create_entity_bone);
        load_kernel(Kernel.create_hull_bone);
        load_kernel(Kernel.create_bone_bind_pose);
        load_kernel(Kernel.create_bone_channel);
        load_kernel(Kernel.create_bone_reference);
        load_kernel(Kernel.create_edge);
        load_kernel(Kernel.create_hull);
        load_kernel(Kernel.create_keyframe);
        load_kernel(Kernel.create_mesh_face);
        load_kernel(Kernel.create_mesh_reference);
        load_kernel(Kernel.create_model_transform);
        load_kernel(Kernel.create_point);
        load_kernel(Kernel.create_texture_uv);
        load_kernel(Kernel.create_vertex_reference);
        load_kernel(Kernel.count_egress_entities);
        load_kernel(Kernel.egress_entities);
        load_kernel(Kernel.merge_point);
        load_kernel(Kernel.merge_edge);
        load_kernel(Kernel.merge_hull);
        load_kernel(Kernel.merge_entity);
        load_kernel(Kernel.merge_hull_bone);
        load_kernel(Kernel.merge_entity_bone);
        load_kernel(Kernel.read_position);
        load_kernel(Kernel.rotate_hull);
        load_kernel(Kernel.set_bone_channel_table);
        load_kernel(Kernel.update_accel);
        load_kernel(Kernel.update_mouse_position);

        return this;
    }
}
