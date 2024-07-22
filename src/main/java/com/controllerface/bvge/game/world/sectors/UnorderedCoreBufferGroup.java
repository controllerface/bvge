package com.controllerface.bvge.game.world.sectors;

import com.controllerface.bvge.cl.buffers.CoreBufferGroup;

import static com.controllerface.bvge.cl.CLData.*;
import static com.controllerface.bvge.cl.buffers.CoreBufferType.*;

public class UnorderedCoreBufferGroup extends CoreBufferGroup
{
    public UnorderedCoreBufferGroup(String name, long ptr_queue, long entity_init, long hull_init, long edge_init, long point_init)
    {
        super(name, ptr_queue, entity_init, hull_init, edge_init, point_init);
    }

    public void unload_sectors(Raw raw, int[] counts)
    {
        int entity_capacity = counts[0];
        int hull_capacity = counts[1];
        int point_capacity = counts[2];
        int edge_capacity = counts[3];
        int hull_bone_capacity = counts[4];
        int entity_bone_capacity = counts[5];

        if (hull_bone_capacity > 0)
        {
            int hull_bone_vec16 = hull_bone_capacity * 16;
            buffer(HULL_BONE).transfer_out_float(raw.hull_bone, cl_float.size(), hull_bone_vec16);
            buffer(HULL_BONE_BIND_POSE).transfer_out_int(raw.hull_bone_bind_pose_id, cl_int.size(), hull_bone_capacity);
            buffer(HULL_BONE_INV_BIND_POSE).transfer_out_int(raw.hull_bone_inv_bind_pose_id, cl_int.size(), hull_bone_capacity);
        }

        if (entity_bone_capacity > 0)
        {
            int entity_bone_vec16 = entity_bone_capacity * 16;
            buffer(ENTITY_BONE).transfer_out_float(raw.entity_bone, cl_float.size(), entity_bone_vec16);
            buffer(ENTITY_BONE_REFERENCE_ID).transfer_out_int(raw.entity_bone_reference_id, cl_int.size(), entity_bone_capacity);
            buffer(ENTITY_BONE_PARENT_ID).transfer_out_int(raw.entity_bone_parent_id, cl_int.size(), entity_bone_capacity);
        }

        if (edge_capacity > 0)
        {
            int edge_vec2 = edge_capacity * 2;
            buffer(EDGE).transfer_out_int(raw.edge, cl_int.size(), edge_vec2);
            buffer(EDGE_LENGTH).transfer_out_float(raw.edge_length, cl_float.size(), edge_capacity);
            buffer(EDGE_FLAG).transfer_out_int(raw.edge_flag, cl_int.size(), edge_capacity);
            buffer(EDGE_PIN).transfer_out_int(raw.edge_pin, cl_int.size(), edge_capacity);
        }

        if (point_capacity > 0)
        {
            int point_vec4 = point_capacity * 4;
            buffer(POINT).transfer_out_float(raw.point, cl_float.size(), point_vec4);
            buffer(POINT_VERTEX_REFERENCE).transfer_out_int(raw.point_vertex_reference, cl_int.size(), point_capacity);
            buffer(POINT_HULL_INDEX).transfer_out_int(raw.point_hull_index, cl_int.size(), point_capacity);
            buffer(POINT_HIT_COUNT).transfer_out_short(raw.point_hit_count, cl_short.size(), point_capacity);
            buffer(POINT_BONE_TABLE).transfer_out_int(raw.point_bone_table, cl_int.size(), point_vec4);
            buffer(POINT_FLAG).transfer_out_int(raw.point_flag, cl_int.size(), point_capacity);
        }

        if (hull_capacity > 0)
        {
            int hull_vec2 = hull_capacity * 2;
            int hull_vec4 = hull_capacity * 4;
            buffer(HULL).transfer_out_float(raw.hull, cl_float.size(), hull_vec4);
            buffer(HULL_SCALE).transfer_out_float(raw.hull_scale, cl_float.size(), hull_vec2);
            buffer(HULL_MESH_ID).transfer_out_int(raw.hull_mesh_id, cl_int.size(), hull_capacity);
            buffer(HULL_UV_OFFSET).transfer_out_int(raw.hull_uv_offset, cl_int.size(), hull_capacity);
            buffer(HULL_ROTATION).transfer_out_float(raw.hull_rotation, cl_float.size(), hull_vec2);
            buffer(HULL_INTEGRITY).transfer_out_int(raw.hull_integrity, cl_int.size(), hull_capacity);
            buffer(HULL_POINT_TABLE).transfer_out_int(raw.hull_point_table, cl_int.size(), hull_vec2);
            buffer(HULL_EDGE_TABLE).transfer_out_int(raw.hull_edge_table, cl_int.size(), hull_vec2);
            buffer(HULL_FLAG).transfer_out_int(raw.hull_flag, cl_int.size(), hull_capacity);
            buffer(HULL_BONE_TABLE).transfer_out_int(raw.hull_bone_table, cl_int.size(), hull_vec2);
            buffer(HULL_ENTITY_ID).transfer_out_int(raw.hull_entity_id, cl_int.size(), hull_capacity);
            buffer(HULL_FRICTION).transfer_out_float(raw.hull_friction, cl_float.size(), hull_capacity);
            buffer(HULL_RESTITUTION).transfer_out_float(raw.hull_restitution, cl_float.size(), hull_capacity);
        }

        if (entity_capacity > 0)
        {
            int entity_vec2 = entity_capacity * 2;
            int entity_vec4 = entity_capacity * 4;
            buffer(ENTITY).transfer_out_float(raw.entity, cl_float.size(), entity_vec4);
            buffer(ENTITY_TYPE).transfer_out_int(raw.entity_type, cl_int.size(), entity_capacity);
            buffer(ENTITY_FLAG).transfer_out_int(raw.entity_flag, cl_int.size(), entity_capacity);
            buffer(ENTITY_ROOT_HULL).transfer_out_int(raw.entity_root_hull, cl_int.size(), entity_capacity);
            buffer(ENTITY_MODEL_ID).transfer_out_int(raw.entity_model_id, cl_int.size(), entity_capacity);
            buffer(ENTITY_TRANSFORM_ID).transfer_out_int(raw.entity_model_transform, cl_int.size(), entity_capacity);
            buffer(ENTITY_MASS).transfer_out_float(raw.entity_mass, cl_float.size(), entity_capacity);
            buffer(ENTITY_ANIM_LAYER).transfer_out_int(raw.entity_anim_layers, cl_int.size(), entity_vec4);
            buffer(ENTITY_PREV_LAYER).transfer_out_int(raw.entity_anim_previous, cl_int.size(), entity_vec4);
            buffer(ENTITY_ANIM_TIME).transfer_out_float(raw.entity_anim_time, cl_float.size(), entity_vec4);
            buffer(ENTITY_PREV_TIME).transfer_out_float(raw.entity_prev_time, cl_float.size(), entity_vec4);
            buffer(ENTITY_MOTION_STATE).transfer_out_short(raw.entity_motion_state, cl_short.size(), entity_vec2);
            buffer(ENTITY_HULL_TABLE).transfer_out_int(raw.entity_hull_table, cl_int.size(), entity_vec2);
            buffer(ENTITY_BONE_TABLE).transfer_out_int(raw.entity_bone_table, cl_int.size(), entity_vec2);
        }
    }

    public static class Raw
    {
        public float[] point = new float[0];
        public int[] point_bone_table = new int[0];
        public int[] point_vertex_reference = new int[0];
        public int[] point_hull_index = new int[0];
        public short[] point_hit_count = new short[0];
        public int[] point_flag = new int[0];

        public int[] edge = new int[0];
        public int[] edge_flag = new int[0];
        public float[] edge_length = new float[0];
        public int[] edge_pin = new int[0];

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
        public float[] entity_anim_time = new float[0];
        public float[] entity_prev_time = new float[0];
        public short[] entity_motion_state = new short[0];
        public int[] entity_anim_layers = new int[0];
        public int[] entity_anim_previous = new int[0];
        public int[] entity_type = new int[0];
        public int[] entity_flag = new int[0];
        public int[] entity_root_hull = new int[0];
        public int[] entity_model_id = new int[0];
        public int[] entity_model_transform = new int[0];
        public int[] entity_hull_table = new int[0];
        public int[] entity_bone_table = new int[0];
        public float[] entity_mass = new float[0];

        public float[] hull_bone = new float[0];
        public int[] hull_bone_bind_pose_id = new int[0];
        public int[] hull_bone_inv_bind_pose_id = new int[0];

        public float[] entity_bone = new float[0];
        public int[] entity_bone_reference_id = new int[0];
        public int[] entity_bone_parent_id = new int[0];

        public void ensure_space(int[] counts)
        {
            int entity_capacity = counts[0];
            int hull_capacity = counts[1];
            int point_capacity = counts[2];
            int edge_capacity = counts[3];
            int hull_bone_capacity = counts[4];
            int entity_bone_capacity = counts[5];

            int entity_vec2 = entity_capacity * 2;
            int entity_vec4 = entity_capacity * 4;
            int hull_vec2 = hull_capacity * 2;
            int hull_vec4 = hull_capacity * 4;
            int point_vec4 = point_capacity * 4;
            int edge_vec2 = edge_capacity * 2;
            int hull_bone_vec16 = hull_bone_capacity * 16;
            int entity_bone_vec16 = entity_bone_capacity * 16;

            entity = ensure_float(entity, entity_vec4);
            entity_anim_time = ensure_float(entity_anim_time, entity_vec4);
            entity_prev_time = ensure_float(entity_prev_time, entity_vec4);
            entity_motion_state = ensure_short(entity_motion_state, entity_vec2);
            entity_anim_layers = ensure_int(entity_anim_layers, entity_vec4);
            entity_anim_previous = ensure_int(entity_anim_previous, entity_vec4);
            entity_flag = ensure_int(entity_flag, entity_capacity);
            entity_type = ensure_int(entity_type, entity_capacity);
            entity_root_hull = ensure_int(entity_root_hull, entity_capacity);
            entity_model_id = ensure_int(entity_model_id, entity_capacity);
            entity_model_transform = ensure_int(entity_model_transform, entity_capacity);
            entity_hull_table = ensure_int(entity_hull_table, entity_vec2);
            entity_bone_table = ensure_int(entity_bone_table, entity_vec2);
            entity_mass = ensure_float(entity_mass, entity_capacity);

            hull = ensure_float(hull, hull_vec4);
            hull_scale = ensure_float(hull_scale, hull_vec2);
            hull_point_table = ensure_int(hull_point_table, hull_vec2);
            hull_edge_table = ensure_int(hull_edge_table, hull_vec2);
            hull_flag = ensure_int(hull_flag, hull_capacity);
            hull_bone_table = ensure_int(hull_bone_table, hull_vec2);
            hull_entity_id = ensure_int(hull_entity_id, hull_capacity);
            hull_friction = ensure_float(hull_friction, hull_capacity);
            hull_restitution = ensure_float(hull_restitution, hull_capacity);
            hull_mesh_id = ensure_int(hull_mesh_id, hull_capacity);
            hull_uv_offset = ensure_int(hull_uv_offset, hull_capacity);
            hull_rotation = ensure_float(hull_rotation, hull_vec2);
            hull_integrity = ensure_int(hull_integrity, hull_capacity);

            point = ensure_float(point, point_vec4);
            point_bone_table = ensure_int(point_bone_table, point_vec4);
            point_vertex_reference = ensure_int(point_vertex_reference, point_capacity);
            point_hull_index = ensure_int(point_hull_index, point_capacity);
            point_flag = ensure_int(point_flag, point_capacity);
            point_hit_count = ensure_short(point_hit_count, point_capacity);

            edge = ensure_int(edge, edge_vec2);
            edge_flag = ensure_int(edge_flag, edge_capacity);
            edge_length = ensure_float(edge_length, edge_capacity);
            edge_pin = ensure_int(edge_pin, edge_capacity);

            hull_bone = ensure_float(hull_bone, hull_bone_vec16);
            hull_bone_bind_pose_id = ensure_int(hull_bone_bind_pose_id, hull_bone_capacity);
            hull_bone_inv_bind_pose_id = ensure_int(hull_bone_inv_bind_pose_id, hull_bone_capacity);

            entity_bone = ensure_float(entity_bone, entity_bone_vec16);
            entity_bone_reference_id = ensure_int(entity_bone_reference_id, entity_bone_capacity);
            entity_bone_parent_id = ensure_int(entity_bone_parent_id, entity_bone_capacity);
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
