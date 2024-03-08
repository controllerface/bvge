package com.controllerface.bvge.gpu;

import com.controllerface.bvge.cl.GPU;

public class GPUReferenceMemory
{
    private int hull_index            = 0;
    private int point_index           = 0;
    private int edge_index            = 0;
    private int vertex_ref_index      = 0;
    private int bone_bind_index       = 0;
    private int bone_ref_index        = 0;
    private int bone_index            = 0;
    private int model_transform_index = 0;
    private int armature_bone_index   = 0;
    private int armature_index        = 0;
    private int mesh_index            = 0;
    private int face_index            = 0;
    private int uv_index              = 0;
    private int keyframe_index        = 0;
    private int bone_channel_index    = 0;
    private int animation_index       = 0;

    // index methods

    public int next_animation_index()
    {
        return animation_index;
    }

    public int next_bone_channel()
    {
        return bone_channel_index;
    }

    public int next_keyframe()
    {
        return keyframe_index;
    }

    public int next_model_transform()
    {
        return model_transform_index;
    }

    public int next_uv()
    {
        return uv_index;
    }

    public int next_face()
    {
        return face_index;
    }

    public int next_mesh()
    {
        return mesh_index;
    }

    public int next_armature()
    {
        return armature_index;
    }

    public int next_hull()
    {
        return hull_index;
    }

    public int next_point()
    {
        return point_index;
    }

    public int next_edge()
    {
        return edge_index;
    }

    public int next_vertex_ref()
    {
        return vertex_ref_index;
    }

    public int next_bone_bind()
    {
        return bone_bind_index;
    }

    public int next_bone_ref()
    {
        return bone_ref_index;
    }

    public int next_bone()
    {
        return bone_index;
    }

    public int next_armature_bone()
    {
        return armature_bone_index;
    }


    // creation methods

    public int new_animation_timings(double[] timings)
    {
        GPU.create_animation_timings(next_animation_index(), timings);
        return animation_index++;
    }

    public int new_bone_channel(int anim_timing_index, int[] pos_table, int[] rot_table, int[] scl_table)
    {
        GPU.create_bone_channel(next_bone_channel(), anim_timing_index, pos_table, rot_table, scl_table);
        return bone_channel_index++;
    }

    public int new_keyframe(float[] frame, double time)
    {
        GPU.create_keyframe(next_keyframe(), frame, time);
        return keyframe_index++;
    }

    public int new_texture_uv(float u, float v)
    {
        GPU.create_texture_uv(next_uv(), u, v);
        return uv_index++;
    }

    public int new_edge(int p1, int p2, float l, int flags)
    {
        GPU.create_edge(next_edge(), p1, p2, l, flags);
        return edge_index++;
    }

    public int new_point(float[] position, int[] vertex_table, int[] bone_ids)
    {
        var init_vert = new float[]{position[0], position[1], position[0], position[1]};
        GPU.create_point(next_point(), init_vert, vertex_table, bone_ids);
        return point_index++;
    }

    public int new_hull(int mesh_id, float[] transform, float[] rotation, int[] table, int[] flags)
    {
        GPU.create_hull(next_hull(), mesh_id, transform, rotation, table, flags);
        return hull_index++;
    }

    public int new_mesh_reference(int[] mesh_ref_table)
    {
        GPU.create_mesh_reference(next_mesh(), mesh_ref_table);
        return mesh_index++;
    }

    public int new_mesh_face(int[] face)
    {
        GPU.create_mesh_face(next_face(), face);
        return face_index++;
    }

    public int new_armature(float x, float y, int[] table, int[] flags, float mass, int anim_index, double anim_time)
    {
        GPU.create_armature(next_armature(), x, y, table, flags, mass, anim_index, anim_time);
        return armature_index++;
    }

    public int new_vertex_reference(float x, float y, float[] weights, int[] uv_table)
    {
        GPU.create_vertex_reference(next_vertex_ref(), x, y, weights, uv_table);
        return vertex_ref_index++;
    }

    public int new_bone_bind_pose(int bind_parent, float[] bone_data)
    {
        GPU.create_bone_bind_pose(next_bone_bind(), bind_parent, bone_data);
        return bone_bind_index++;
    }

    public int new_bone_reference(float[] bone_data)
    {
        GPU.create_bone_reference(next_bone_ref(), bone_data);
        return bone_ref_index++;
    }

    public int new_bone(int[] offset_id, float[] bone_data)
    {
        GPU.create_bone(next_bone(), offset_id, bone_data);
        return bone_index++;
    }

    public int new_armature_bone(int[] bone_bind_table, float[] bone_data)
    {
        GPU.create_armature_bone(next_armature_bone(), bone_bind_table, bone_data);
        return armature_bone_index++;
    }

    public int new_model_transform(float[] transform_data)
    {
        GPU.create_model_transform(next_model_transform(), transform_data);
        return model_transform_index++;
    }

    public void compact_buffers(int edge_shift,
                                       int bone_shift,
                                       int point_shift,
                                       int hull_shift,
                                       int armature_shift,
                                       int armature_bone_shift)
    {
        edge_index          -= (edge_shift);
        bone_index          -= (bone_shift);
        point_index         -= (point_shift);
        hull_index          -= (hull_shift);
        armature_index      -= (armature_shift);
        armature_bone_index -= (armature_bone_shift);
    }
}
