package com.controllerface.bvge.game.world.sectors;

public interface ReferenceContainer
{
    int next_mesh();
    int new_vertex_reference(float x, float y, float[] weights, int[] uv_table);
    int new_bone_bind_pose(float[] bone_data);
    int new_bone_reference(float[] bone_data);
    int new_animation_timings(float duration, float tick_rate);
    int new_bone_channel(int anim_timing_index, int[] pos_table, int[] rot_table, int[] scl_table);
    int new_keyframe(float[] frame, float time);
    int new_texture_uv(float u, float v);
    int new_mesh_reference(int[] vertex_table, int[] face_table);
    int new_mesh_face(int[] face);
    int new_model_transform(float[] transform_data);
    void set_bone_channel_table(int bind_pose_target, int[] channel_table);
}
