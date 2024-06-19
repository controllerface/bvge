package com.controllerface.bvge.game.world.sectors;

import com.controllerface.bvge.cl.buffers.BufferGroup;

import static com.controllerface.bvge.cl.CLSize.*;
import static com.controllerface.bvge.cl.buffers.BufferType.*;

public class SectorGroup extends BufferGroup
{
    public SectorGroup(String name, long ptr_queue, long entity_init, long hull_init, long edge_init, long point_init)
    {
        super(name, ptr_queue);

        //#region Point Buffers

        /* int4
         * x: bone 1 instance id
         * y: bone 2 instance id
         * z: bone 3 instance id
         * w: bone 4 instance id
         */
        set_buffer(POINT_BONE_TABLE, cl_int4, point_init);

        /* float4
         * x: current x position
         * y: current y position
         * z: previous x position
         * w: previous y position
         */
        set_buffer(POINT, cl_float4, point_init);

        /* int
         * x: reference vertex index
         */
        set_buffer(POINT_VERTEX_REFERENCE, cl_int, point_init);

        /* int
         * x: hull index
         */
        set_buffer(POINT_HULL_INDEX, cl_int, point_init);

        /* int
         * x: vertex flags (bit field)
         */
        set_buffer(POINT_FLAG, cl_int, point_init);

        /* ushort
         * x: recent collision hit counter
         */
        set_buffer(POINT_HIT_COUNT, cl_short, point_init);

        /* float
         * x: anti-gravity magnitude for each point
         */
        set_buffer(POINT_ANTI_GRAV, cl_float, point_init);

        //#endregion

        //#region Edge Buffers

        /* int2
         * x: point 1 index
         * y: point 2 index
         */
        set_buffer(EDGE, cl_int2, edge_init);

        /* float
         * x: edge constraint length
         */
        set_buffer(EDGE_LENGTH, cl_float, edge_init);

        /* int
         * x: edge flags (bit-field)
         */
        set_buffer(EDGE_FLAG, cl_int, edge_init);

        //#endregion

        //#region Hull Buffers

        /* float4
         * x: current x position
         * y: current y position
         * z: previous x position
         * w: previous y position
         */
        set_buffer(HULL, cl_float4, hull_init);

        /* float2
         * x: scale x
         * y: scale y
         */
        set_buffer(HULL_SCALE, cl_float2, hull_init);

        /* int2
         * x: start point index
         * y: end point index
         */
        set_buffer(HULL_POINT_TABLE, cl_int2, hull_init);

        /* int2
         * x: start edge index
         * y: end edge index
         */
        set_buffer(HULL_EDGE_TABLE, cl_int2, hull_init);

        /* int
         * x: hull flags (bit-field)
         */
        set_buffer(HULL_FLAG, cl_int, hull_init);

        /* int
         * x: entity id for aligned hull
         */
        set_buffer(HULL_ENTITY_ID, cl_int, hull_init);

        /* int2
         * x: start bone
         * y: end bone
         */
        set_buffer(HULL_BONE_TABLE, cl_int2, hull_init);

        /* float
         * x: friction coefficient
         */
        set_buffer(HULL_FRICTION, cl_float, hull_init);

        /* float
         * x: restitution coefficient
         */
        set_buffer(HULL_RESTITUTION, cl_float, hull_init);

        /* int
         * x: reference mesh id
         */
        set_buffer(HULL_MESH_ID, cl_int, hull_init);

        /* int
         * x: offset index of the UV to use for this hull
         */
        set_buffer(HULL_UV_OFFSET, cl_int, hull_init);

        /* float2
         * x: initial reference angle
         * y: current rotation
         */
        set_buffer(HULL_ROTATION, cl_float2, hull_init);

        /* int
         * x: the integrity (i.e. health) of the hull
         */
        set_buffer(HULL_INTEGRITY, cl_int, hull_init);

        /* float4
         * x: corner x position
         * y: corner y position
         * z: width
         * w: height
         */
        set_buffer(HULL_AABB, cl_float4, hull_init);

        /* int4
         * x: minimum x key index
         * y: maximum x key index
         * z: minimum y key index
         * w: maximum y key index
         */
        set_buffer(HULL_AABB_INDEX, cl_int4, hull_init);

        /* int2
         * x: key bank offset
         * y: key bank size
         */
        set_buffer(HULL_AABB_KEY_TABLE, cl_int2, hull_init);

        //#endregion

        //#region Entity Buffers

        /* float2
         * x: the last rendered timestamp of the current animation
         * y: the last rendered timestamp of the previous animation
         */
        set_buffer(ENTITY_ANIM_ELAPSED, cl_float2, entity_init);

        /* short2
         * x: number of ticks moving downward
         * y: number of ticks moving upward
         */
        set_buffer(ENTITY_MOTION_STATE, cl_short2, entity_init);

        /* int2
         * x: the currently running animation index
         * y: the previously running animation index
         */
        set_buffer(ENTITY_ANIM_INDEX, cl_int2, entity_init);

        /* float4
         * x: current x position
         * y: current y position
         * z: previous x position
         * w: previous y position
         */
        set_buffer(ENTITY, cl_float4, entity_init);

        /* int
         * x: unique entity type id
         */
        set_buffer(ENTITY_TYPE, cl_int, entity_init);

        /* int
         * x: entity flags (bit-field)
         */
        set_buffer(ENTITY_FLAG, cl_int, entity_init);

        /* int
         * x: root hull index of the aligned entity
         */
        set_buffer(ENTITY_ROOT_HULL, cl_int, entity_init);

        /* int
         * x: model id of the aligned entity
         */
        set_buffer(ENTITY_MODEL_ID, cl_int, entity_init);

        /* int
         * x: model transform index of the aligned entity
         */
        set_buffer(ENTITY_TRANSFORM_ID, cl_int, entity_init);

        /* int2
         * x: start hull index
         * y: end hull index
         */
        set_buffer(ENTITY_HULL_TABLE, cl_int2, entity_init);

        /* int2
         * x: start bone anim index
         * y: end bone anim index
         */
        set_buffer(ENTITY_BONE_TABLE, cl_int2, entity_init);

        /* float
         * x: mass of the entity
         */
        set_buffer(ENTITY_MASS, cl_float, entity_init);

        /* float2
         * x: current x acceleration
         * y: current y acceleration
         */
        set_buffer(ENTITY_ACCEL, cl_float2, entity_init);

        /* float2
         * x: the initial time of the current blend operation
         * y: the remaining time of the current blend operation
         */
        set_buffer(ENTITY_ANIM_BLEND, cl_float2, entity_init);

        //#endregion

        //#region Hull Bone Buffers

        /* float16
         * s0-sF: Column-major, 4x4 transformation matrix, hull bone instance
         */
        set_buffer(HULL_BONE, cl_float16, hull_init);

        /* int
         * x: bone bind pose index (model space)
         */
        set_buffer(HULL_BONE_BIND_POSE, cl_int, hull_init);

        /* int
         * x: bone inverse bind pose index (mesh-space)
         */
        set_buffer(HULL_BONE_INV_BIND_POSE, cl_int, hull_init);

        //#endregion

        //#region Armature Bone Buffers

        /* float16
         * s0-sF: Column-major, 4x4 transformation matrix, armature bone instance
         */
        set_buffer(ENTITY_BONE, cl_float16, entity_init);

        /* int
         * x: bind pose reference id
         */
        set_buffer(ENTITY_BONE_REFERENCE_ID, cl_int, entity_init);

        /* int
         * x: armature bone parent id
         */
        set_buffer(ENTITY_BONE_PARENT_ID, cl_int, entity_init);

        //#endregion
    }

    public void ensure_capacity(int point_capacity, int edge_capacity, int hull_capacity, int entity_capacity, int hull_bone_capacity, int entity_bone_capacity)
    {
        get_buffer(HULL_BONE).ensure_capacity(hull_bone_capacity);
        get_buffer(HULL_BONE_BIND_POSE).ensure_capacity(hull_bone_capacity);
        get_buffer(HULL_BONE_INV_BIND_POSE).ensure_capacity(hull_bone_capacity);

        get_buffer(ENTITY_BONE).ensure_capacity(entity_bone_capacity);
        get_buffer(ENTITY_BONE_REFERENCE_ID).ensure_capacity(entity_bone_capacity);
        get_buffer(ENTITY_BONE_PARENT_ID).ensure_capacity(entity_bone_capacity);

        get_buffer(EDGE).ensure_capacity(edge_capacity);
        get_buffer(EDGE_LENGTH).ensure_capacity(edge_capacity);
        get_buffer(EDGE_FLAG).ensure_capacity(edge_capacity);

        get_buffer(POINT).ensure_capacity(point_capacity);
        get_buffer(POINT_VERTEX_REFERENCE).ensure_capacity(point_capacity);
        get_buffer(POINT_HULL_INDEX).ensure_capacity(point_capacity);
        get_buffer(POINT_BONE_TABLE).ensure_capacity(point_capacity);
        get_buffer(POINT_HIT_COUNT).ensure_capacity(point_capacity);
        get_buffer(POINT_FLAG).ensure_capacity(point_capacity);
        get_buffer(POINT_ANTI_GRAV).ensure_capacity(point_capacity);

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
}
