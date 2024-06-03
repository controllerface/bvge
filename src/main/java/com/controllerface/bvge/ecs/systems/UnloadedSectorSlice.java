package com.controllerface.bvge.ecs.systems;

public class UnloadedSectorSlice
{
    public int[] raw_point_bone_table           = new int[0];
    public float[] raw_point                    = new float[0];
    public int[] raw_point_vertex_reference     = new int[0];
    public int[] raw_point_hull_index           = new int[0];
    public int[] raw_point_flag                 = new int[0];
    public short[] raw_point_hit_count          = new short[0];

    public int[] raw_edge                       = new int[0];
    public int[] raw_edge_flag                  = new int[0];
    public float[] raw_edge_length              = new float[0];

    public float[] raw_hull                     = new float[0];
    public float[] raw_hull_scale               = new float[0];
    public int[] raw_hull_point_table           = new int[0];
    public int[] raw_hull_edge_table            = new int[0];
    public int[] raw_hull_flag                  = new int[0];
    public int[] raw_hull_bone_table            = new int[0];
    public int[] raw_hull_entity_id             = new int[0];
    public float[] raw_hull_friction            = new float[0];
    public float[] raw_hull_restitution         = new float[0];
    public int[] raw_hull_mesh_id               = new int[0];
    public int[] raw_hull_uv_offset             = new int[0];
    public float[] raw_hull_rotation            = new float[0];
    public int[] raw_hull_integrity             = new int[0];

    public float[] raw_entity_anim_elapsed      = new float[0];
    public short[] raw_entity_motion_state      = new short[0];
    public int[] raw_entity_anim_index          = new int[0];
    public float[] raw_entity                   = new float[0];
    public int[] raw_entity_flag                = new int[0];
    public int[] raw_entity_root_hull           = new int[0];
    public int[] raw_entity_model_id            = new int[0];
    public int[] raw_entity_model_transform     = new int[0];
    public int[] raw_entity_hull_table          = new int[0];
    public int[] raw_entity_bone_table          = new int[0];
    public float[] raw_entity_mass              = new float[0];

    public float[] raw_hull_bone                = new float[0];
    public int[] raw_hull_bone_bind_pose_id     = new int[0];
    public int[] raw_hull_bone_inv_bind_pose_id = new int[0];

    public float[] raw_entity_bone              = new float[0];
    public int[] raw_entity_bone_reference_id   = new int[0];
    public int[] raw_entity_bone_parent_id      = new int[0];

    public void ensure_space(int[] counts)
    {
        int entity_capacity        = counts[0];
        int hull_capacity          = counts[1];
        int point_capacity         = counts[2];
        int edge_capacity          = counts[3];
        int hull_bone_capacity     = counts[4];
        int entity_bone_capacity   = counts[5];

        int entity_vec2       = entity_capacity      * 2;
        int entity_vec4       = entity_capacity      * 4;
        int hull_vec2         = hull_capacity        * 2;
        int hull_vec4         = hull_capacity        * 4;
        int point_vec4        = point_capacity       * 4;
        int edge_vec2         = edge_capacity        * 2;
        int hull_bone_vec16   = hull_bone_capacity   * 16;
        int entity_bone_vec16 = entity_bone_capacity * 16;

        raw_entity                     = ensure_float(raw_entity,                   entity_vec4);
        raw_entity_anim_elapsed        = ensure_float(raw_entity_anim_elapsed,      entity_vec2);
        raw_entity_motion_state        = ensure_short(raw_entity_motion_state,      entity_vec2);
        raw_entity_anim_index          = ensure_int(raw_entity_anim_index,          entity_vec2);
        raw_entity_flag                = ensure_int(raw_entity_flag,                entity_capacity);
        raw_entity_root_hull           = ensure_int(raw_entity_root_hull,           entity_capacity);
        raw_entity_model_id            = ensure_int(raw_entity_model_id,            entity_capacity);
        raw_entity_model_transform     = ensure_int(raw_entity_model_transform,     entity_capacity);
        raw_entity_hull_table          = ensure_int(raw_entity_hull_table,          entity_vec2);
        raw_entity_bone_table          = ensure_int(raw_entity_bone_table,          entity_vec2);
        raw_entity_mass                = ensure_float(raw_entity_mass,              entity_capacity);

        raw_hull                       = ensure_float(raw_hull,                     hull_vec4);
        raw_hull_scale                 = ensure_float(raw_hull_scale,               hull_vec2);
        raw_hull_point_table           = ensure_int(raw_hull_point_table,           hull_vec2);
        raw_hull_edge_table            = ensure_int(raw_hull_edge_table,            hull_vec2);
        raw_hull_flag                  = ensure_int(raw_hull_flag,                  hull_capacity);
        raw_hull_bone_table            = ensure_int(raw_hull_bone_table,            hull_vec2);
        raw_hull_entity_id             = ensure_int(raw_hull_entity_id,             hull_capacity);
        raw_hull_friction              = ensure_float(raw_hull_friction,            hull_capacity);
        raw_hull_restitution           = ensure_float(raw_hull_restitution,         hull_capacity);
        raw_hull_mesh_id               = ensure_int(raw_hull_mesh_id,               hull_capacity);
        raw_hull_uv_offset             = ensure_int(raw_hull_uv_offset,             hull_capacity);
        raw_hull_rotation              = ensure_float(raw_hull_rotation,            hull_vec2);
        raw_hull_integrity             = ensure_int(raw_hull_integrity,             hull_capacity);

        raw_point                      = ensure_float(raw_point,                    point_vec4);
        raw_point_bone_table           = ensure_int(raw_point_bone_table,           point_vec4);
        raw_point_vertex_reference     = ensure_int(raw_point_vertex_reference,     point_capacity);
        raw_point_hull_index           = ensure_int(raw_point_hull_index,           point_capacity);
        raw_point_flag                 = ensure_int(raw_point_flag,                 point_capacity);
        raw_point_hit_count            = ensure_short(raw_point_hit_count,          point_capacity);

        raw_edge                       = ensure_int(raw_edge,                       edge_vec2);
        raw_edge_flag                  = ensure_int(raw_edge_flag,                  edge_capacity);
        raw_edge_length                = ensure_float(raw_edge_length,              edge_capacity);

        raw_hull_bone                  = ensure_float(raw_hull_bone,                hull_bone_vec16);
        raw_hull_bone_bind_pose_id     = ensure_int(raw_hull_bone_bind_pose_id,     hull_bone_capacity);
        raw_hull_bone_inv_bind_pose_id = ensure_int(raw_hull_bone_inv_bind_pose_id, hull_bone_capacity);

        raw_entity_bone                = ensure_float(raw_entity_bone,              entity_bone_vec16);
        raw_entity_bone_reference_id   = ensure_int(raw_entity_bone_reference_id,   entity_bone_capacity);
        raw_entity_bone_parent_id      = ensure_int(raw_entity_bone_parent_id,      entity_bone_capacity);
    }

    private float[] ensure_float(float[] input, int required_capacity)
    {
        return input.length >= required_capacity
            ? input
            : new float[required_capacity];
    }

    private int[] ensure_int(int[] input, int required_capacity)
    {
        return input.length >= required_capacity
            ? input
            : new int[required_capacity];
    }

    private short[] ensure_short(short[] input, int required_capacity)
    {
        return input.length >= required_capacity
            ? input
            : new short[required_capacity];
    }
}
