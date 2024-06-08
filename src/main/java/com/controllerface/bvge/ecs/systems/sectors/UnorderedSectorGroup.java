package com.controllerface.bvge.ecs.systems.sectors;

import com.controllerface.bvge.cl.CLSize;
import com.controllerface.bvge.cl.buffers.BufferGroup;
import com.controllerface.bvge.cl.buffers.BufferType;

public class UnorderedSectorGroup extends BufferGroup
{
    public UnorderedSectorGroup(long ptr_queue, long entity_init, long hull_init, long edge_init, long point_init)
    {
        super(ptr_queue);

        set_buffer(BufferType.EDGE,                     new_buffer(CLSize.cl_int2,    edge_init));
        set_buffer(BufferType.EDGE_LENGTH,              new_buffer(CLSize.cl_float,   edge_init));
        set_buffer(BufferType.EDGE_FLAG,                new_buffer(CLSize.cl_int,     edge_init));

        set_buffer(BufferType.HULL_BONE,                new_buffer(CLSize.cl_float16, hull_init));
        set_buffer(BufferType.HULL_BONE_BIND_POSE,      new_buffer(CLSize.cl_int,     hull_init));
        set_buffer(BufferType.HULL_BONE_INV_BIND_POSE,  new_buffer(CLSize.cl_int,     hull_init));

        set_buffer(BufferType.ENTITY_BONE,              new_buffer(CLSize.cl_float16));
        set_buffer(BufferType.ENTITY_BONE_REFERENCE_ID, new_buffer(CLSize.cl_int));
        set_buffer(BufferType.ENTITY_BONE_PARENT_ID,    new_buffer(CLSize.cl_int));

        set_buffer(BufferType.POINT,                    new_buffer(CLSize.cl_float4, point_init));
        set_buffer(BufferType.POINT_VERTEX_REFERENCE,   new_buffer(CLSize.cl_int,    point_init));
        set_buffer(BufferType.POINT_HULL_INDEX,         new_buffer(CLSize.cl_int,    point_init));
        set_buffer(BufferType.POINT_BONE_TABLE,         new_buffer(CLSize.cl_int4,   point_init));
        set_buffer(BufferType.POINT_HIT_COUNT,          new_buffer(CLSize.cl_short,  point_init));
        set_buffer(BufferType.POINT_FLAG,               new_buffer(CLSize.cl_int,    point_init));

        set_buffer(BufferType.HULL,                     new_buffer(CLSize.cl_float4, hull_init));
        set_buffer(BufferType.HULL_SCALE,               new_buffer(CLSize.cl_float2, hull_init));
        set_buffer(BufferType.HULL_POINT_TABLE,         new_buffer(CLSize.cl_int2,   hull_init));
        set_buffer(BufferType.HULL_EDGE_TABLE,          new_buffer(CLSize.cl_int2,   hull_init));
        set_buffer(BufferType.HULL_FLAG,                new_buffer(CLSize.cl_int,    hull_init));
        set_buffer(BufferType.HULL_BONE_TABLE,          new_buffer(CLSize.cl_int2,   hull_init));
        set_buffer(BufferType.HULL_ENTITY_ID,           new_buffer(CLSize.cl_int,    hull_init));
        set_buffer(BufferType.HULL_FRICTION,            new_buffer(CLSize.cl_float,  hull_init));
        set_buffer(BufferType.HULL_RESTITUTION,         new_buffer(CLSize.cl_float,  hull_init));
        set_buffer(BufferType.HULL_MESH_ID,             new_buffer(CLSize.cl_int,    hull_init));
        set_buffer(BufferType.HULL_UV_OFFSET,           new_buffer(CLSize.cl_int,    hull_init));
        set_buffer(BufferType.HULL_ROTATION,            new_buffer(CLSize.cl_float2, hull_init));
        set_buffer(BufferType.HULL_INTEGRITY,           new_buffer(CLSize.cl_int,    hull_init));

        set_buffer(BufferType.ENTITY_ANIM_ELAPSED,      new_buffer(CLSize.cl_float2, entity_init));
        set_buffer(BufferType.ENTITY_MOTION_STATE,      new_buffer(CLSize.cl_short2, entity_init));
        set_buffer(BufferType.ENTITY_ANIM_INDEX,        new_buffer(CLSize.cl_int2,   entity_init));
        set_buffer(BufferType.ENTITY,                   new_buffer(CLSize.cl_float4, entity_init));
        set_buffer(BufferType.ENTITY_FLAG,              new_buffer(CLSize.cl_int,    entity_init));
        set_buffer(BufferType.ENTITY_ROOT_HULL,         new_buffer(CLSize.cl_int,    entity_init));
        set_buffer(BufferType.ENTITY_MODEL_ID,          new_buffer(CLSize.cl_int,    entity_init));
        set_buffer(BufferType.ENTITY_TRANSFORM_ID,      new_buffer(CLSize.cl_int,    entity_init));
        set_buffer(BufferType.ENTITY_HULL_TABLE,        new_buffer(CLSize.cl_int2,   entity_init));
        set_buffer(BufferType.ENTITY_BONE_TABLE,        new_buffer(CLSize.cl_int2,   entity_init));
        set_buffer(BufferType.ENTITY_MASS,              new_buffer(CLSize.cl_float,  entity_init));
    }

    public void ensure_capacity(int[] egress_counts)
    {
        int entity_capacity        = egress_counts[0];
        int hull_capacity          = egress_counts[1];
        int point_capacity         = egress_counts[2];
        int edge_capacity          = egress_counts[3];
        int hull_bone_capacity     = egress_counts[4];
        int entity_bone_capacity   = egress_counts[5];

        buffer(BufferType.HULL_BONE).ensure_capacity(hull_bone_capacity);
        buffer(BufferType.HULL_BONE_BIND_POSE).ensure_capacity(hull_bone_capacity);
        buffer(BufferType.HULL_BONE_INV_BIND_POSE).ensure_capacity(hull_bone_capacity);

        buffer(BufferType.ENTITY_BONE).ensure_capacity(entity_bone_capacity);
        buffer(BufferType.ENTITY_BONE_REFERENCE_ID).ensure_capacity(entity_bone_capacity);
        buffer(BufferType.ENTITY_BONE_PARENT_ID).ensure_capacity(entity_bone_capacity);

        buffer(BufferType.EDGE).ensure_capacity(edge_capacity);
        buffer(BufferType.EDGE_LENGTH).ensure_capacity(edge_capacity);
        buffer(BufferType.EDGE_FLAG).ensure_capacity(edge_capacity);

        buffer(BufferType.POINT).ensure_capacity(point_capacity);
        buffer(BufferType.POINT_VERTEX_REFERENCE).ensure_capacity(point_capacity);
        buffer(BufferType.POINT_HULL_INDEX).ensure_capacity(point_capacity);
        buffer(BufferType.POINT_BONE_TABLE).ensure_capacity(point_capacity);
        buffer(BufferType.POINT_HIT_COUNT).ensure_capacity(point_capacity);
        buffer(BufferType.POINT_FLAG).ensure_capacity(point_capacity);

        buffer(BufferType.ENTITY_ANIM_ELAPSED).ensure_capacity(entity_capacity);
        buffer(BufferType.ENTITY_MOTION_STATE).ensure_capacity(entity_capacity);
        buffer(BufferType.ENTITY_ANIM_INDEX).ensure_capacity(entity_capacity);
        buffer(BufferType.ENTITY).ensure_capacity(entity_capacity);
        buffer(BufferType.ENTITY_FLAG).ensure_capacity(entity_capacity);
        buffer(BufferType.ENTITY_ROOT_HULL).ensure_capacity(entity_capacity);
        buffer(BufferType.ENTITY_MODEL_ID).ensure_capacity(entity_capacity);
        buffer(BufferType.ENTITY_TRANSFORM_ID).ensure_capacity(entity_capacity);
        buffer(BufferType.ENTITY_HULL_TABLE).ensure_capacity(entity_capacity);
        buffer(BufferType.ENTITY_BONE_TABLE).ensure_capacity(entity_capacity);
        buffer(BufferType.ENTITY_MASS).ensure_capacity(entity_capacity);

        buffer(BufferType.HULL).ensure_capacity(hull_capacity);
        buffer(BufferType.HULL_SCALE).ensure_capacity(hull_capacity);
        buffer(BufferType.HULL_POINT_TABLE).ensure_capacity(hull_capacity);
        buffer(BufferType.HULL_EDGE_TABLE).ensure_capacity(hull_capacity);
        buffer(BufferType.HULL_FLAG).ensure_capacity(hull_capacity);
        buffer(BufferType.HULL_BONE_TABLE).ensure_capacity(hull_capacity);
        buffer(BufferType.HULL_ENTITY_ID).ensure_capacity(hull_capacity);
        buffer(BufferType.HULL_FRICTION).ensure_capacity(hull_capacity);
        buffer(BufferType.HULL_RESTITUTION).ensure_capacity(hull_capacity);
        buffer(BufferType.HULL_MESH_ID).ensure_capacity(hull_capacity);
        buffer(BufferType.HULL_UV_OFFSET).ensure_capacity(hull_capacity);
        buffer(BufferType.HULL_ROTATION).ensure_capacity(hull_capacity);
        buffer(BufferType.HULL_INTEGRITY).ensure_capacity(hull_capacity);
    }

    public void unload_sectors(Raw raw_sectors, int[] counts)
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

        if (point_capacity > 0)
        {
            buffer(BufferType.POINT).transfer_out_float(raw_sectors.raw_point,                                         CLSize.cl_float,  point_vec4);
            buffer(BufferType.POINT_VERTEX_REFERENCE).transfer_out_int(raw_sectors.raw_point_vertex_reference,         CLSize.cl_int,     point_capacity);
            buffer(BufferType.POINT_HULL_INDEX).transfer_out_int(raw_sectors.raw_point_hull_index,                     CLSize.cl_int,     point_capacity);
            buffer(BufferType.POINT_HIT_COUNT).transfer_out_short(raw_sectors.raw_point_hit_count,                     CLSize.cl_short,   point_capacity);
            buffer(BufferType.POINT_BONE_TABLE).transfer_out_int(raw_sectors.raw_point_bone_table,                     CLSize.cl_int,    point_vec4);
            buffer(BufferType.POINT_FLAG).transfer_out_int(raw_sectors.raw_point_flag,                                 CLSize.cl_int,     point_capacity);
        }

        if (edge_capacity > 0)
        {
            buffer(BufferType.EDGE).transfer_out_int(raw_sectors.raw_edge,                                             CLSize.cl_int,    edge_vec2);
            buffer(BufferType.EDGE_LENGTH).transfer_out_float(raw_sectors.raw_edge_length,                             CLSize.cl_float,   edge_capacity);
            buffer(BufferType.EDGE_FLAG).transfer_out_int(raw_sectors.raw_edge_flag,                                   CLSize.cl_int,     edge_capacity);
        }

        if (hull_capacity > 0)
        {
            buffer(BufferType.HULL).transfer_out_float(raw_sectors.raw_hull,                                           CLSize.cl_float,  hull_vec4);
            buffer(BufferType.HULL_SCALE).transfer_out_float(raw_sectors.raw_hull_scale,                               CLSize.cl_float,  hull_vec2);
            buffer(BufferType.HULL_MESH_ID).transfer_out_int(raw_sectors.raw_hull_mesh_id,                             CLSize.cl_int,     hull_capacity);
            buffer(BufferType.HULL_UV_OFFSET).transfer_out_int(raw_sectors.raw_hull_uv_offset,                         CLSize.cl_int,     hull_capacity);
            buffer(BufferType.HULL_ROTATION).transfer_out_float(raw_sectors.raw_hull_rotation,                         CLSize.cl_float,  hull_vec2);
            buffer(BufferType.HULL_INTEGRITY).transfer_out_int(raw_sectors.raw_hull_integrity,                         CLSize.cl_int,     hull_capacity);
            buffer(BufferType.HULL_POINT_TABLE).transfer_out_int(raw_sectors.raw_hull_point_table,                     CLSize.cl_int,    hull_vec2);
            buffer(BufferType.HULL_EDGE_TABLE).transfer_out_int(raw_sectors.raw_hull_edge_table,                       CLSize.cl_int,    hull_vec2);
            buffer(BufferType.HULL_FLAG).transfer_out_int(raw_sectors.raw_hull_flag,                                   CLSize.cl_int,     hull_capacity);
            buffer(BufferType.HULL_BONE_TABLE).transfer_out_int(raw_sectors.raw_hull_bone_table,                       CLSize.cl_int,    hull_vec2);
            buffer(BufferType.HULL_ENTITY_ID).transfer_out_int(raw_sectors.raw_hull_entity_id,                         CLSize.cl_int,     hull_capacity);
            buffer(BufferType.HULL_FRICTION).transfer_out_float(raw_sectors.raw_hull_friction,                         CLSize.cl_float,   hull_capacity);
            buffer(BufferType.HULL_RESTITUTION).transfer_out_float(raw_sectors.raw_hull_restitution,                   CLSize.cl_float,   hull_capacity);
        }

        if (entity_capacity > 0)
        {
            buffer(BufferType.ENTITY).transfer_out_float(raw_sectors.raw_entity,                                       CLSize.cl_float,  entity_vec4);
            buffer(BufferType.ENTITY_FLAG).transfer_out_int(raw_sectors.raw_entity_flag,                               CLSize.cl_int,     entity_capacity);
            buffer(BufferType.ENTITY_ROOT_HULL).transfer_out_int(raw_sectors.raw_entity_root_hull,                     CLSize.cl_int,     entity_capacity);
            buffer(BufferType.ENTITY_MODEL_ID).transfer_out_int(raw_sectors.raw_entity_model_id,                       CLSize.cl_int,     entity_capacity);
            buffer(BufferType.ENTITY_TRANSFORM_ID).transfer_out_int(raw_sectors.raw_entity_model_transform,            CLSize.cl_int,     entity_capacity);
            buffer(BufferType.ENTITY_MASS).transfer_out_float(raw_sectors.raw_entity_mass,                             CLSize.cl_float,   entity_capacity);
            buffer(BufferType.ENTITY_ANIM_INDEX).transfer_out_int(raw_sectors.raw_entity_anim_index,                   CLSize.cl_int,    entity_vec2);
            buffer(BufferType.ENTITY_ANIM_ELAPSED).transfer_out_float(raw_sectors.raw_entity_anim_elapsed,             CLSize.cl_float,  entity_vec2);
            buffer(BufferType.ENTITY_MOTION_STATE).transfer_out_short(raw_sectors.raw_entity_motion_state,             CLSize.cl_short,  entity_vec2);
            buffer(BufferType.ENTITY_HULL_TABLE).transfer_out_int(raw_sectors.raw_entity_hull_table,                   CLSize.cl_int,    entity_vec2);
            buffer(BufferType.ENTITY_BONE_TABLE).transfer_out_int(raw_sectors.raw_entity_bone_table,                   CLSize.cl_int,    entity_vec2);
        }

        if (hull_bone_capacity > 0)
        {
            buffer(BufferType.HULL_BONE).transfer_out_float(raw_sectors.raw_hull_bone,                                 CLSize.cl_float,  hull_bone_vec16);
            buffer(BufferType.HULL_BONE_BIND_POSE).transfer_out_int(raw_sectors.raw_hull_bone_bind_pose_id,            CLSize.cl_int,    hull_bone_capacity);
            buffer(BufferType.HULL_BONE_INV_BIND_POSE).transfer_out_int(raw_sectors.raw_hull_bone_inv_bind_pose_id,    CLSize.cl_int,    hull_bone_capacity);
        }

        if (entity_bone_capacity > 0)
        {
            buffer(BufferType.ENTITY_BONE).transfer_out_float(raw_sectors.raw_entity_bone,                             CLSize.cl_float,  entity_bone_vec16);
            buffer(BufferType.ENTITY_BONE_REFERENCE_ID).transfer_out_int(raw_sectors.raw_entity_bone_reference_id,     CLSize.cl_int,    entity_bone_capacity);
            buffer(BufferType.ENTITY_BONE_PARENT_ID).transfer_out_int(raw_sectors.raw_entity_bone_parent_id,           CLSize.cl_int,    entity_bone_capacity);
        }
    }

    public static class Raw
    {
        public float[] raw_point                    = new float[0];
        public int[]   raw_point_bone_table           = new int[0];
        public int[]   raw_point_vertex_reference   = new int[0];
        public int[]   raw_point_hull_index         = new int[0];
        public short[] raw_point_hit_count          = new short[0];
        public int[]   raw_point_flag               = new int[0];

        public int[] raw_edge                       = new int[0];
        public int[] raw_edge_flag                  = new int[0];
        public float[] raw_edge_length              = new float[0];

        public float[] raw_hull                     = new float[0];
        public float[] raw_hull_scale               = new float[0];
        public int[] raw_hull_point_table           = new int[0];
        public int[] raw_hull_edge_table            = new int[0];
        public int[] raw_hull_bone_table            = new int[0];
        public float[] raw_hull_rotation            = new float[0];
        public int[] raw_hull_flag                  = new int[0];
        public int[] raw_hull_entity_id             = new int[0];
        public float[] raw_hull_friction            = new float[0];
        public float[] raw_hull_restitution         = new float[0];
        public int[] raw_hull_mesh_id               = new int[0];
        public int[] raw_hull_uv_offset             = new int[0];
        public int[] raw_hull_integrity             = new int[0];

        public float[] raw_entity                   = new float[0];
        public float[] raw_entity_anim_elapsed      = new float[0];
        public short[] raw_entity_motion_state      = new short[0];
        public int[] raw_entity_anim_index          = new int[0];
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
}
