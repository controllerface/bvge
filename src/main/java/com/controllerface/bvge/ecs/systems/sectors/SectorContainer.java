package com.controllerface.bvge.ecs.systems.sectors;

public interface SectorContainer
{
    int next_point();
    int next_edge();
    int next_hull();
    int next_entity();
    int next_hull_bone();
    int next_armature_bone();

    int new_point(float[] position, int[] bone_ids, int vertex_index, int hull_index, int hit_count, int flags);
    int new_edge(int p1, int p2, float l, int flags);
    int new_hull(int mesh_id,
                 float[] position,
                 float[] scale,
                 float[] rotation,
                 int[] point_table,
                 int[] edge_table,
                 int[] bone_table,
                 float friction,
                 float restitution,
                 int entity_id,
                 int uv_offset,
                 int flags);
    int new_entity(float x, float y, float z, float w,
                   int[] hull_table,
                   int[] bone_table,
                   float mass,
                   int anim_index,
                   float anim_time,
                   int root_hull,
                   int model_id,
                   int model_transform_id,
                   int type,
                   int flags);
    int new_hull_bone(float[] bone_data, int bind_pose_id, int inv_bind_pose_id);
    int new_armature_bone(int bone_reference, int bone_parent_id, float[] bone_data);

    void destroy();
}
