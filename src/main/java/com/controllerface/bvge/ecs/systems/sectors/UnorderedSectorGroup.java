package com.controllerface.bvge.ecs.systems.sectors;

import com.controllerface.bvge.cl.CLSize;
import com.controllerface.bvge.cl.buffers.BufferGroup;
import com.controllerface.bvge.cl.buffers.BufferType;

import static com.controllerface.bvge.cl.buffers.BufferType.*;

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

        buffer(HULL_BONE)                .ensure_capacity(hull_bone_capacity);
        buffer(HULL_BONE_BIND_POSE)      .ensure_capacity(hull_bone_capacity);
        buffer(HULL_BONE_INV_BIND_POSE)  .ensure_capacity(hull_bone_capacity);

        buffer(ENTITY_BONE)              .ensure_capacity(entity_bone_capacity);
        buffer(ENTITY_BONE_REFERENCE_ID) .ensure_capacity(entity_bone_capacity);
        buffer(ENTITY_BONE_PARENT_ID)    .ensure_capacity(entity_bone_capacity);

        buffer(EDGE)                     .ensure_capacity(edge_capacity);
        buffer(EDGE_LENGTH)              .ensure_capacity(edge_capacity);
        buffer(EDGE_FLAG)                .ensure_capacity(edge_capacity);

        buffer(POINT)                    .ensure_capacity(point_capacity);
        buffer(POINT_VERTEX_REFERENCE)   .ensure_capacity(point_capacity);
        buffer(POINT_HULL_INDEX)         .ensure_capacity(point_capacity);
        buffer(POINT_BONE_TABLE)         .ensure_capacity(point_capacity);
        buffer(POINT_HIT_COUNT)          .ensure_capacity(point_capacity);
        buffer(POINT_FLAG)               .ensure_capacity(point_capacity);

        buffer(HULL)                     .ensure_capacity(hull_capacity);
        buffer(HULL_SCALE)               .ensure_capacity(hull_capacity);
        buffer(HULL_POINT_TABLE)         .ensure_capacity(hull_capacity);
        buffer(HULL_EDGE_TABLE)          .ensure_capacity(hull_capacity);
        buffer(HULL_FLAG)                .ensure_capacity(hull_capacity);
        buffer(HULL_BONE_TABLE)          .ensure_capacity(hull_capacity);
        buffer(HULL_ENTITY_ID)           .ensure_capacity(hull_capacity);
        buffer(HULL_FRICTION)            .ensure_capacity(hull_capacity);
        buffer(HULL_RESTITUTION)         .ensure_capacity(hull_capacity);
        buffer(HULL_MESH_ID)             .ensure_capacity(hull_capacity);
        buffer(HULL_UV_OFFSET)           .ensure_capacity(hull_capacity);
        buffer(HULL_ROTATION)            .ensure_capacity(hull_capacity);
        buffer(HULL_INTEGRITY)           .ensure_capacity(hull_capacity);

        buffer(ENTITY_ANIM_ELAPSED)      .ensure_capacity(entity_capacity);
        buffer(ENTITY_MOTION_STATE)      .ensure_capacity(entity_capacity);
        buffer(ENTITY_ANIM_INDEX)        .ensure_capacity(entity_capacity);
        buffer(ENTITY)                   .ensure_capacity(entity_capacity);
        buffer(ENTITY_FLAG)              .ensure_capacity(entity_capacity);
        buffer(ENTITY_ROOT_HULL)         .ensure_capacity(entity_capacity);
        buffer(ENTITY_MODEL_ID)          .ensure_capacity(entity_capacity);
        buffer(ENTITY_TRANSFORM_ID)      .ensure_capacity(entity_capacity);
        buffer(ENTITY_HULL_TABLE)        .ensure_capacity(entity_capacity);
        buffer(ENTITY_BONE_TABLE)        .ensure_capacity(entity_capacity);
        buffer(ENTITY_MASS)              .ensure_capacity(entity_capacity);
    }

    public void unload_sectors(Raw raw, int[] counts)
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

        if (hull_bone_capacity > 0)
        {
            buffer(HULL_BONE).transfer_out_float(raw.hull_bone,                              CLSize.cl_float,  hull_bone_vec16);
            buffer(HULL_BONE_BIND_POSE).transfer_out_int(raw.hull_bone_bind_pose_id,         CLSize.cl_int,    hull_bone_capacity);
            buffer(HULL_BONE_INV_BIND_POSE).transfer_out_int(raw.hull_bone_inv_bind_pose_id, CLSize.cl_int,    hull_bone_capacity);
        }

        if (entity_bone_capacity > 0)
        {
            buffer(ENTITY_BONE).transfer_out_float(raw.entity_bone,                          CLSize.cl_float,  entity_bone_vec16);
            buffer(ENTITY_BONE_REFERENCE_ID).transfer_out_int(raw.entity_bone_reference_id,  CLSize.cl_int,    entity_bone_capacity);
            buffer(ENTITY_BONE_PARENT_ID).transfer_out_int(raw.entity_bone_parent_id,        CLSize.cl_int,    entity_bone_capacity);
        }

        if (edge_capacity > 0)
        {
            buffer(EDGE).transfer_out_int(raw.edge,                                          CLSize.cl_int,    edge_vec2);
            buffer(EDGE_LENGTH).transfer_out_float(raw.edge_length,                          CLSize.cl_float,   edge_capacity);
            buffer(EDGE_FLAG).transfer_out_int(raw.edge_flag,                                CLSize.cl_int,     edge_capacity);
        }

        if (point_capacity > 0)
        {
            buffer(POINT).transfer_out_float(raw.point,                                      CLSize.cl_float,  point_vec4);
            buffer(POINT_VERTEX_REFERENCE).transfer_out_int(raw.point_vertex_reference,      CLSize.cl_int,     point_capacity);
            buffer(POINT_HULL_INDEX).transfer_out_int(raw.point_hull_index,                  CLSize.cl_int,     point_capacity);
            buffer(POINT_HIT_COUNT).transfer_out_short(raw.point_hit_count,                  CLSize.cl_short,   point_capacity);
            buffer(POINT_BONE_TABLE).transfer_out_int(raw.point_bone_table,                  CLSize.cl_int,    point_vec4);
            buffer(POINT_FLAG).transfer_out_int(raw.point_flag,                              CLSize.cl_int,     point_capacity);
        }

        if (hull_capacity > 0)
        {
            buffer(HULL).transfer_out_float(raw.hull,                                        CLSize.cl_float,  hull_vec4);
            buffer(HULL_SCALE).transfer_out_float(raw.hull_scale,                            CLSize.cl_float,  hull_vec2);
            buffer(HULL_MESH_ID).transfer_out_int(raw.hull_mesh_id,                          CLSize.cl_int,     hull_capacity);
            buffer(HULL_UV_OFFSET).transfer_out_int(raw.hull_uv_offset,                      CLSize.cl_int,     hull_capacity);
            buffer(HULL_ROTATION).transfer_out_float(raw.hull_rotation,                      CLSize.cl_float,  hull_vec2);
            buffer(HULL_INTEGRITY).transfer_out_int(raw.hull_integrity,                      CLSize.cl_int,     hull_capacity);
            buffer(HULL_POINT_TABLE).transfer_out_int(raw.hull_point_table,                  CLSize.cl_int,    hull_vec2);
            buffer(HULL_EDGE_TABLE).transfer_out_int(raw.hull_edge_table,                    CLSize.cl_int,    hull_vec2);
            buffer(HULL_FLAG).transfer_out_int(raw.hull_flag,                                CLSize.cl_int,     hull_capacity);
            buffer(HULL_BONE_TABLE).transfer_out_int(raw.hull_bone_table,                    CLSize.cl_int,    hull_vec2);
            buffer(HULL_ENTITY_ID).transfer_out_int(raw.hull_entity_id,                      CLSize.cl_int,     hull_capacity);
            buffer(HULL_FRICTION).transfer_out_float(raw.hull_friction,                      CLSize.cl_float,   hull_capacity);
            buffer(HULL_RESTITUTION).transfer_out_float(raw.hull_restitution,                CLSize.cl_float,   hull_capacity);
        }

        if (entity_capacity > 0)
        {
            buffer(ENTITY).transfer_out_float(raw.entity,                                    CLSize.cl_float,  entity_vec4);
            buffer(ENTITY_FLAG).transfer_out_int(raw.entity_flag,                            CLSize.cl_int,     entity_capacity);
            buffer(ENTITY_ROOT_HULL).transfer_out_int(raw.entity_root_hull,                  CLSize.cl_int,     entity_capacity);
            buffer(ENTITY_MODEL_ID).transfer_out_int(raw.entity_model_id,                    CLSize.cl_int,     entity_capacity);
            buffer(ENTITY_TRANSFORM_ID).transfer_out_int(raw.entity_model_transform,         CLSize.cl_int,     entity_capacity);
            buffer(ENTITY_MASS).transfer_out_float(raw.entity_mass,                          CLSize.cl_float,   entity_capacity);
            buffer(ENTITY_ANIM_INDEX).transfer_out_int(raw.entity_anim_index,                CLSize.cl_int,    entity_vec2);
            buffer(ENTITY_ANIM_ELAPSED).transfer_out_float(raw.entity_anim_elapsed,          CLSize.cl_float,  entity_vec2);
            buffer(ENTITY_MOTION_STATE).transfer_out_short(raw.entity_motion_state,          CLSize.cl_short,  entity_vec2);
            buffer(ENTITY_HULL_TABLE).transfer_out_int(raw.entity_hull_table,                CLSize.cl_int,    entity_vec2);
            buffer(ENTITY_BONE_TABLE).transfer_out_int(raw.entity_bone_table,                CLSize.cl_int,    entity_vec2);
        }
    }

    public static class Raw
    {
        public float[] hull_bone = new float[0];
        public int[] hull_bone_bind_pose_id = new int[0];
        public int[] hull_bone_inv_bind_pose_id = new int[0];

        public float[] entity_bone = new float[0];
        public int[] entity_bone_reference_id = new int[0];
        public int[] entity_bone_parent_id = new int[0];

        public int[] edge = new int[0];
        public int[] edge_flag = new int[0];
        public float[] edge_length = new float[0];

        public float[] point = new float[0];
        public int[] point_bone_table = new int[0];
        public int[] point_vertex_reference = new int[0];
        public int[] point_hull_index = new int[0];
        public short[] point_hit_count = new short[0];
        public int[] point_flag = new int[0];

        public float[] hull = new float[0];
        public float[] hull_scale = new float[0];
        public int[] hull_point_table = new int[0];
        public int[] hull_edge_table = new int[0];
        public int[] hull_bone_table = new int[0];
        public float[] hull_rotation = new float[0];
        public int[] hull_flag = new int[0];
        public int[] hull_entity_id = new int[0];
        public float[] hull_friction = new float[0];
        public float[] hull_restitution = new float[0];
        public int[] hull_mesh_id = new int[0];
        public int[] hull_uv_offset = new int[0];
        public int[] hull_integrity = new int[0];

        public float[] entity = new float[0];
        public float[] entity_anim_elapsed = new float[0];
        public short[] entity_motion_state = new short[0];
        public int[] entity_anim_index = new int[0];
        public int[] entity_flag = new int[0];
        public int[] entity_root_hull = new int[0];
        public int[] entity_model_id = new int[0];
        public int[] entity_model_transform = new int[0];
        public int[] entity_hull_table = new int[0];
        public int[] entity_bone_table = new int[0];
        public float[] entity_mass = new float[0];

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

            entity = ensure_float(entity,                   entity_vec4);
            entity_anim_elapsed = ensure_float(entity_anim_elapsed,      entity_vec2);
            entity_motion_state = ensure_short(entity_motion_state,      entity_vec2);
            entity_anim_index = ensure_int(entity_anim_index,          entity_vec2);
            entity_flag = ensure_int(entity_flag,                entity_capacity);
            entity_root_hull = ensure_int(entity_root_hull,           entity_capacity);
            entity_model_id = ensure_int(entity_model_id,            entity_capacity);
            entity_model_transform = ensure_int(entity_model_transform,     entity_capacity);
            entity_hull_table = ensure_int(entity_hull_table,          entity_vec2);
            entity_bone_table = ensure_int(entity_bone_table,          entity_vec2);
            entity_mass = ensure_float(entity_mass,              entity_capacity);

            hull = ensure_float(hull,                     hull_vec4);
            hull_scale = ensure_float(hull_scale,               hull_vec2);
            hull_point_table = ensure_int(hull_point_table,           hull_vec2);
            hull_edge_table = ensure_int(hull_edge_table,            hull_vec2);
            hull_flag = ensure_int(hull_flag,                  hull_capacity);
            hull_bone_table = ensure_int(hull_bone_table,            hull_vec2);
            hull_entity_id = ensure_int(hull_entity_id,             hull_capacity);
            hull_friction = ensure_float(hull_friction,            hull_capacity);
            hull_restitution = ensure_float(hull_restitution,         hull_capacity);
            hull_mesh_id = ensure_int(hull_mesh_id,               hull_capacity);
            hull_uv_offset = ensure_int(hull_uv_offset,             hull_capacity);
            hull_rotation = ensure_float(hull_rotation,            hull_vec2);
            hull_integrity = ensure_int(hull_integrity,             hull_capacity);

            point = ensure_float(point,                    point_vec4);
            point_bone_table = ensure_int(point_bone_table,           point_vec4);
            point_vertex_reference = ensure_int(point_vertex_reference,     point_capacity);
            point_hull_index = ensure_int(point_hull_index,           point_capacity);
            point_flag = ensure_int(point_flag,                 point_capacity);
            point_hit_count = ensure_short(point_hit_count,          point_capacity);

            edge = ensure_int(edge,                       edge_vec2);
            edge_flag = ensure_int(edge_flag,                  edge_capacity);
            edge_length = ensure_float(edge_length,              edge_capacity);

            hull_bone = ensure_float(hull_bone,                hull_bone_vec16);
            hull_bone_bind_pose_id = ensure_int(hull_bone_bind_pose_id,     hull_bone_capacity);
            hull_bone_inv_bind_pose_id = ensure_int(hull_bone_inv_bind_pose_id, hull_bone_capacity);

            entity_bone = ensure_float(entity_bone,              entity_bone_vec16);
            entity_bone_reference_id = ensure_int(entity_bone_reference_id,   entity_bone_capacity);
            entity_bone_parent_id = ensure_int(entity_bone_parent_id,      entity_bone_capacity);
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
