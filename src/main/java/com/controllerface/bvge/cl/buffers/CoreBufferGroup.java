package com.controllerface.bvge.cl.buffers;

import static com.controllerface.bvge.cl.CLSize.*;
import static com.controllerface.bvge.cl.buffers.CoreBufferType.*;

public class CoreBufferGroup extends BufferGroup<CoreBufferType>
{
    public CoreBufferGroup(String name, long ptr_queue, long entity_init, long hull_init, long edge_init, long point_init)
    {
        super(CoreBufferType.class, name, ptr_queue, true);

        //#region Point Buffers

        set_buffer(POINT_BONE_TABLE,       point_init);
        set_buffer(POINT,                  point_init);
        set_buffer(POINT_VERTEX_REFERENCE, point_init);
        set_buffer(POINT_HULL_INDEX,       point_init);
        set_buffer(POINT_FLAG,             point_init);
        set_buffer(POINT_HIT_COUNT,        point_init);
        set_buffer(POINT_ANTI_GRAV,        point_init);
        set_buffer(POINT_ANTI_TIME,        point_init);

        //#endregion

        //#region Edge Buffers

        set_buffer(EDGE,                edge_init);
        set_buffer(EDGE_LENGTH,         edge_init);
        set_buffer(EDGE_FLAG,           edge_init);
        set_buffer(EDGE_AABB,           edge_init);
        set_buffer(EDGE_AABB_INDEX,     edge_init);
        set_buffer(EDGE_AABB_KEY_TABLE, edge_init);

        //#endregion

        //#region Hull Buffers


        set_buffer(HULL,                hull_init);
        set_buffer(HULL_SCALE,          hull_init);
        set_buffer(HULL_POINT_TABLE,    hull_init);
        set_buffer(HULL_EDGE_TABLE,     hull_init);
        set_buffer(HULL_FLAG,           hull_init);
        set_buffer(HULL_ENTITY_ID,      hull_init);
        set_buffer(HULL_BONE_TABLE,     hull_init);
        set_buffer(HULL_FRICTION,       hull_init);
        set_buffer(HULL_RESTITUTION,    hull_init);
        set_buffer(HULL_MESH_ID,        hull_init);
        set_buffer(HULL_UV_OFFSET,      hull_init);
        set_buffer(HULL_ROTATION,       hull_init);
        set_buffer(HULL_INTEGRITY,      hull_init);
        set_buffer(HULL_AABB,           hull_init);
        set_buffer(HULL_AABB_INDEX,     hull_init);
        set_buffer(HULL_AABB_KEY_TABLE, hull_init);

        //#endregion

        //#region Entity Buffers

        set_buffer(ENTITY_ANIM_ELAPSED, entity_init);
        set_buffer(ENTITY_MOTION_STATE, entity_init);
        set_buffer(ENTITY_ANIM_INDEX,   entity_init);
        set_buffer(ENTITY,              entity_init);
        set_buffer(ENTITY_TYPE,         entity_init);
        set_buffer(ENTITY_FLAG,         entity_init);
        set_buffer(ENTITY_ROOT_HULL,    entity_init);
        set_buffer(ENTITY_MODEL_ID,     entity_init);
        set_buffer(ENTITY_TRANSFORM_ID, entity_init);
        set_buffer(ENTITY_HULL_TABLE,   entity_init);
        set_buffer(ENTITY_BONE_TABLE,   entity_init);
        set_buffer(ENTITY_MASS,         entity_init);
        set_buffer(ENTITY_ACCEL,        entity_init);
        set_buffer(ENTITY_ANIM_BLEND,   entity_init);

        //#endregion

        //#region Hull Bone Buffers

        set_buffer(HULL_BONE,               hull_init);
        set_buffer(HULL_BONE_BIND_POSE,     hull_init);
        set_buffer(HULL_BONE_INV_BIND_POSE, hull_init);

        //#endregion

        //#region Armature Bone Buffers

        set_buffer(ENTITY_BONE,              entity_init);
        set_buffer(ENTITY_BONE_REFERENCE_ID, entity_init);
        set_buffer(ENTITY_BONE_PARENT_ID,    entity_init);

        //#endregion
    }

    public void ensure_capacity_all(int point_capacity,
                                    int edge_capacity,
                                    int hull_capacity,
                                    int entity_capacity,
                                    int hull_bone_capacity,
                                    int entity_bone_capacity)
    {
        ensure_point_capacity(point_capacity);
        ensure_edge_capacity(edge_capacity);
        ensure_hull_capacity(hull_capacity);
        ensure_entity_capacity(entity_capacity);
        ensure_hull_bone_capacity(hull_bone_capacity);
        ensure_entity_bone_capacity(entity_bone_capacity);
    }

    public void ensure_point_capacity(int point_capacity)
    {
        get_buffer(POINT).ensure_capacity(point_capacity);
        get_buffer(POINT_VERTEX_REFERENCE).ensure_capacity(point_capacity);
        get_buffer(POINT_HULL_INDEX).ensure_capacity(point_capacity);
        get_buffer(POINT_BONE_TABLE).ensure_capacity(point_capacity);
        get_buffer(POINT_HIT_COUNT).ensure_capacity(point_capacity);
        get_buffer(POINT_FLAG).ensure_capacity(point_capacity);
        get_buffer(POINT_ANTI_GRAV).ensure_capacity(point_capacity);
        get_buffer(POINT_ANTI_TIME).ensure_capacity(point_capacity);
    }

    public void ensure_edge_capacity(int edge_capacity)
    {
        get_buffer(EDGE).ensure_capacity(edge_capacity);
        get_buffer(EDGE_LENGTH).ensure_capacity(edge_capacity);
        get_buffer(EDGE_FLAG).ensure_capacity(edge_capacity);
        get_buffer(EDGE_AABB).ensure_capacity(edge_capacity);
        get_buffer(EDGE_AABB_INDEX).ensure_capacity(edge_capacity);
        get_buffer(EDGE_AABB_KEY_TABLE).ensure_capacity(edge_capacity);
    }

    public void ensure_hull_capacity(int hull_capacity)
    {
        get_buffer(HULL).ensure_capacity(hull_capacity);
        get_buffer(HULL_SCALE).ensure_capacity(hull_capacity);
        get_buffer(HULL_POINT_TABLE).ensure_capacity(hull_capacity);
        get_buffer(HULL_EDGE_TABLE).ensure_capacity(hull_capacity);
        get_buffer(HULL_FLAG).ensure_capacity(hull_capacity);
        get_buffer(HULL_BONE_TABLE).ensure_capacity(hull_capacity);
        get_buffer(HULL_ENTITY_ID).ensure_capacity(hull_capacity);
        get_buffer(HULL_FRICTION).ensure_capacity(hull_capacity);
        get_buffer(HULL_RESTITUTION).ensure_capacity(hull_capacity);
        get_buffer(HULL_MESH_ID).ensure_capacity(hull_capacity);
        get_buffer(HULL_UV_OFFSET).ensure_capacity(hull_capacity);
        get_buffer(HULL_ROTATION).ensure_capacity(hull_capacity);
        get_buffer(HULL_INTEGRITY).ensure_capacity(hull_capacity);
        get_buffer(HULL_AABB).ensure_capacity(hull_capacity);
        get_buffer(HULL_AABB_INDEX).ensure_capacity(hull_capacity);
        get_buffer(HULL_AABB_KEY_TABLE).ensure_capacity(hull_capacity);
    }

    public void ensure_entity_capacity(int entity_capacity)
    {
        get_buffer(ENTITY).ensure_capacity(entity_capacity);
        get_buffer(ENTITY_ANIM_ELAPSED).ensure_capacity(entity_capacity);
        get_buffer(ENTITY_MOTION_STATE).ensure_capacity(entity_capacity);
        get_buffer(ENTITY_ANIM_INDEX).ensure_capacity(entity_capacity);
        get_buffer(ENTITY_TYPE).ensure_capacity(entity_capacity);
        get_buffer(ENTITY_FLAG).ensure_capacity(entity_capacity);
        get_buffer(ENTITY_ROOT_HULL).ensure_capacity(entity_capacity);
        get_buffer(ENTITY_MODEL_ID).ensure_capacity(entity_capacity);
        get_buffer(ENTITY_TRANSFORM_ID).ensure_capacity(entity_capacity);
        get_buffer(ENTITY_HULL_TABLE).ensure_capacity(entity_capacity);
        get_buffer(ENTITY_BONE_TABLE).ensure_capacity(entity_capacity);
        get_buffer(ENTITY_MASS).ensure_capacity(entity_capacity);
        get_buffer(ENTITY_ACCEL).ensure_capacity(entity_capacity);
        get_buffer(ENTITY_ANIM_BLEND).ensure_capacity(entity_capacity);
    }

    public void ensure_hull_bone_capacity(int hull_bone_capacity)
    {
        get_buffer(HULL_BONE).ensure_capacity(hull_bone_capacity);
        get_buffer(HULL_BONE_BIND_POSE).ensure_capacity(hull_bone_capacity);
        get_buffer(HULL_BONE_INV_BIND_POSE).ensure_capacity(hull_bone_capacity);
    }

    public void ensure_entity_bone_capacity(int entity_bone_capacity)
    {
        get_buffer(ENTITY_BONE).ensure_capacity(entity_bone_capacity);
        get_buffer(ENTITY_BONE_REFERENCE_ID).ensure_capacity(entity_bone_capacity);
        get_buffer(ENTITY_BONE_PARENT_ID).ensure_capacity(entity_bone_capacity);
    }
}
