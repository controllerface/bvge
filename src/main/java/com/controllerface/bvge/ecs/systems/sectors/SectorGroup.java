package com.controllerface.bvge.ecs.systems.sectors;

import com.controllerface.bvge.cl.CLSize;
import com.controllerface.bvge.cl.buffers.BufferGroup;
import com.controllerface.bvge.cl.buffers.BufferType;

import static com.controllerface.bvge.cl.buffers.BufferType.*;

public class SectorGroup extends BufferGroup
{
    public SectorGroup(long ptr_queue, long entity_init, long hull_init, long edge_init, long point_init)
    {
        super(ptr_queue);

        //#region Edge Buffers

        /* int2
         * x: point 1 index
         * y: point 2 index
         */
        set_buffer(BufferType.EDGE, new_buffer(CLSize.cl_int2, edge_init));

        /* float
         * x: edge constraint length
         */
        set_buffer(BufferType.EDGE_LENGTH, new_buffer(CLSize.cl_float, edge_init));

        /* int
         * x: edge flags (bit-field)
         */
        set_buffer(BufferType.EDGE_FLAG, new_buffer(CLSize.cl_int, edge_init));

        //#endregion

        //#region Hull Bone Buffers

        /* float16
         * s0-sF: Column-major, 4x4 transformation matrix, hull bone instance
         */
        set_buffer(BufferType.HULL_BONE, new_buffer(CLSize.cl_float16, hull_init));

        /* int
         * x: bone bind pose index (model space)
         */
        set_buffer(BufferType.HULL_BONE_BIND_POSE, new_buffer(CLSize.cl_int, hull_init));

        /* int
         * x: bone inverse bind pose index (mesh-space)
         */
        set_buffer(BufferType.HULL_BONE_INV_BIND_POSE, new_buffer(CLSize.cl_int, hull_init));

        //#endregion

        //#region Armature Bone Buffers

        /* float16
         * s0-sF: Column-major, 4x4 transformation matrix, armature bone instance
         */
        set_buffer(BufferType.ENTITY_BONE, new_buffer(CLSize.cl_float16, entity_init));

        /* int
         * x: bind pose reference id
         */
        set_buffer(BufferType.ENTITY_BONE_REFERENCE_ID, new_buffer(CLSize.cl_int, entity_init));

        /* int
         * x: armature bone parent id
         */
        set_buffer(BufferType.ENTITY_BONE_PARENT_ID, new_buffer(CLSize.cl_int, entity_init));

        //#endregion

        //#region Point Buffers

        /* int4
         * x: bone 1 instance id
         * y: bone 2 instance id
         * z: bone 3 instance id
         * w: bone 4 instance id
         */
        set_buffer(BufferType.POINT_BONE_TABLE, new_buffer(CLSize.cl_int4,   point_init));

        /* float4
         * x: current x position
         * y: current y position
         * z: previous x position
         * w: previous y position
         */
        set_buffer(BufferType.POINT, new_buffer(CLSize.cl_float4, point_init));

        /* int
         * x: reference vertex index
         */
        set_buffer(BufferType.POINT_VERTEX_REFERENCE, new_buffer(CLSize.cl_int, point_init));

        /* int
         * x: hull index
         */
        set_buffer(BufferType.POINT_HULL_INDEX, new_buffer(CLSize.cl_int, point_init));

        /* int
         * x: vertex flags (bit field)
         */
        set_buffer(BufferType.POINT_FLAG, new_buffer(CLSize.cl_int, point_init));

        /* ushort
         * x: recent collision hit counter
         */
        set_buffer(BufferType.POINT_HIT_COUNT, new_buffer(CLSize.cl_short, point_init));

        //#endregion

        //#region Hull Buffers

        /* float4
         * x: current x position
         * y: current y position
         * z: previous x position
         * w: previous y position
         */
        set_buffer(BufferType.HULL, new_buffer(CLSize.cl_float4, hull_init));

        /* float2
         * x: scale x
         * y: scale y
         */
        set_buffer(BufferType.HULL_SCALE, new_buffer(CLSize.cl_float2, hull_init));

        /* int2
         * x: start point index
         * y: end point index
         */
        set_buffer(BufferType.HULL_POINT_TABLE, new_buffer(CLSize.cl_int2, hull_init));

        /* int2
         * x: start edge index
         * y: end edge index
         */
        set_buffer(BufferType.HULL_EDGE_TABLE, new_buffer(CLSize.cl_int2, hull_init));

        /* int
         * x: hull flags (bit-field)
         */
        set_buffer(BufferType.HULL_FLAG, new_buffer(CLSize.cl_int, hull_init));

        /* int
         * x: entity id for aligned hull
         */
        set_buffer(BufferType.HULL_ENTITY_ID, new_buffer(CLSize.cl_int, hull_init));

        /* int2
         * x: start bone
         * y: end bone
         */
        set_buffer(BufferType.HULL_BONE_TABLE, new_buffer(CLSize.cl_int2, hull_init));

        /* float
         * x: friction coefficient
         */
        set_buffer(BufferType.HULL_FRICTION, new_buffer(CLSize.cl_float, hull_init));

        /* float
         * x: restitution coefficient
         */
        set_buffer(BufferType.HULL_RESTITUTION, new_buffer(CLSize.cl_float, hull_init));

        /* int
         * x: reference mesh id
         */
        set_buffer(BufferType.HULL_MESH_ID, new_buffer(CLSize.cl_int, hull_init));

        /* int
         * x: offset index of the UV to use for this hull
         */
        set_buffer(BufferType.HULL_UV_OFFSET, new_buffer(CLSize.cl_int, hull_init));

        /* float2
         * x: initial reference angle
         * y: current rotation
         */
        set_buffer(BufferType.HULL_ROTATION, new_buffer(CLSize.cl_float2, hull_init));

        /* int
         * x: the integrity (i.e. health) of the hull
         */
        set_buffer(BufferType.HULL_INTEGRITY, new_buffer(CLSize.cl_int, hull_init));

        //#endregion

        //#region Entity Buffers

        /* float2
         * x: the last rendered timestamp of the current animation
         * y: the last rendered timestamp of the previous animation
         */
        set_buffer(BufferType.ENTITY_ANIM_ELAPSED, new_buffer(CLSize.cl_float2, entity_init));

        /* short2
         * x: number of ticks moving downward
         * y: number of ticks moving upward
         */
        set_buffer(BufferType.ENTITY_MOTION_STATE, new_buffer(CLSize.cl_short2, entity_init));

        /* int2
         * x: the currently running animation index
         * y: the previously running animation index
         */
        set_buffer(BufferType.ENTITY_ANIM_INDEX, new_buffer(CLSize.cl_int2, entity_init));

        /* float4
         * x: current x position
         * y: current y position
         * z: previous x position
         * w: previous y position
         */
        set_buffer(BufferType.ENTITY, new_buffer(CLSize.cl_float4, entity_init));

        /* int
         * x: entity flags (bit-field)
         */
        set_buffer(BufferType.ENTITY_FLAG, new_buffer(CLSize.cl_int, entity_init));

        /* int
         * x: root hull index of the aligned entity
         */
        set_buffer(BufferType.ENTITY_ROOT_HULL, new_buffer(CLSize.cl_int, entity_init));

        /* int
         * x: model id of the aligned entity
         */
        set_buffer(BufferType.ENTITY_MODEL_ID, new_buffer(CLSize.cl_int, entity_init));

        /* int
         * x: model transform index of the aligned entity
         */
        set_buffer(BufferType.ENTITY_TRANSFORM_ID, new_buffer(CLSize.cl_int, entity_init));

        /* int2
         * x: start hull index
         * y: end hull index
         */
        set_buffer(BufferType.ENTITY_HULL_TABLE, new_buffer(CLSize.cl_int2, entity_init));

        /* int2
         * x: start bone anim index
         * y: end bone anim index
         */
        set_buffer(BufferType.ENTITY_BONE_TABLE, new_buffer(CLSize.cl_int2, entity_init));

        /* float
         * x: mass of the entity
         */
        set_buffer(BufferType.ENTITY_MASS, new_buffer(CLSize.cl_float, entity_init));

        //#endregion
    }

    public void ensure_capacity(int point_capacity, int edge_capacity, int hull_capacity, int entity_capacity, int hull_bone_capacity, int entity_bone_capacity)
    {
        buffer(HULL_BONE).ensure_capacity(hull_bone_capacity);
        buffer(HULL_BONE_BIND_POSE).ensure_capacity(hull_bone_capacity);
        buffer(HULL_BONE_INV_BIND_POSE).ensure_capacity(hull_bone_capacity);

        buffer(ENTITY_BONE).ensure_capacity(entity_bone_capacity);
        buffer(ENTITY_BONE_REFERENCE_ID).ensure_capacity(entity_bone_capacity);
        buffer(ENTITY_BONE_PARENT_ID).ensure_capacity(entity_bone_capacity);

        buffer(EDGE).ensure_capacity(edge_capacity);
        buffer(EDGE_LENGTH).ensure_capacity(edge_capacity);
        buffer(EDGE_FLAG).ensure_capacity(edge_capacity);

        buffer(POINT).ensure_capacity(point_capacity);
        buffer(POINT_VERTEX_REFERENCE).ensure_capacity(point_capacity);
        buffer(POINT_HULL_INDEX).ensure_capacity(point_capacity);
        buffer(POINT_BONE_TABLE).ensure_capacity(point_capacity);
        buffer(POINT_HIT_COUNT).ensure_capacity(point_capacity);
        buffer(POINT_FLAG).ensure_capacity(point_capacity);

        buffer(HULL).ensure_capacity(hull_capacity);
        buffer(HULL_SCALE).ensure_capacity(hull_capacity);
        buffer(HULL_POINT_TABLE).ensure_capacity(hull_capacity);
        buffer(HULL_EDGE_TABLE).ensure_capacity(hull_capacity);
        buffer(HULL_FLAG).ensure_capacity(hull_capacity);
        buffer(HULL_BONE_TABLE).ensure_capacity(hull_capacity);
        buffer(HULL_ENTITY_ID).ensure_capacity(hull_capacity);
        buffer(HULL_FRICTION).ensure_capacity(hull_capacity);
        buffer(HULL_RESTITUTION).ensure_capacity(hull_capacity);
        buffer(HULL_MESH_ID).ensure_capacity(hull_capacity);
        buffer(HULL_UV_OFFSET).ensure_capacity(hull_capacity);
        buffer(HULL_ROTATION).ensure_capacity(hull_capacity);
        buffer(HULL_INTEGRITY).ensure_capacity(hull_capacity);

        buffer(ENTITY_ANIM_ELAPSED).ensure_capacity(entity_capacity);
        buffer(ENTITY_MOTION_STATE).ensure_capacity(entity_capacity);
        buffer(ENTITY_ANIM_INDEX).ensure_capacity(entity_capacity);
        buffer(ENTITY).ensure_capacity(entity_capacity);
        buffer(ENTITY_FLAG).ensure_capacity(entity_capacity);
        buffer(ENTITY_ROOT_HULL).ensure_capacity(entity_capacity);
        buffer(ENTITY_MODEL_ID).ensure_capacity(entity_capacity);
        buffer(ENTITY_TRANSFORM_ID).ensure_capacity(entity_capacity);
        buffer(ENTITY_HULL_TABLE).ensure_capacity(entity_capacity);
        buffer(ENTITY_BONE_TABLE).ensure_capacity(entity_capacity);
        buffer(ENTITY_MASS).ensure_capacity(entity_capacity);
    }
}
