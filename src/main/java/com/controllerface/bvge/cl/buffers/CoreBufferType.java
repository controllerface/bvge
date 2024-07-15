package com.controllerface.bvge.cl.buffers;

import static com.controllerface.bvge.cl.CLData.*;

public enum CoreBufferType implements BufferType
{
    //#region Entity Buffers

    /*
     * x: current x position
     * y: current y position
     * z: previous x position
     * w: previous y position
     */
    ENTITY                      (cl_float4),

    /* float2
     * x: current x acceleration
     * y: current y acceleration
     */
    ENTITY_ACCEL                (cl_float2),

    /* float2
     * x: the initial time of the current blend operation
     * y: the remaining time of the current blend operation
     */
    ENTITY_ANIM_BLEND           (cl_float2),

    /*
     * x: the last rendered timestamp of the current animation
     * y: the last rendered timestamp of the previous animation
     */
    ENTITY_ANIM_ELAPSED         (cl_float2),

    /*
     * x: the current layer 0 animation index
     * y: the current layer 1 animation index
     */
    ENTITY_ANIM_LAYER           (cl_int2),

    /*
     * x: the previous layer 0 animation index
     * y: the previous layer 1 animation index
     */
    ENTITY_ANIM_PREVIOUS       (cl_int2),

    /*
     * x: number of ticks moving downward
     * y: number of ticks moving upward
     */
    ENTITY_MOTION_STATE         (cl_short2),

    /*
     * x: unique entity type id
     */
    ENTITY_TYPE                 (cl_int),

    /*
     * x: entity flags (bit-field)
     */
    ENTITY_FLAG                 (cl_int),

    /*
     * x: start bone anim index
     * y: end bone anim index
     */
    ENTITY_BONE_TABLE           (cl_int2),

    /*
     * x: start hull index
     * y: end hull index
     */
    ENTITY_HULL_TABLE           (cl_int2),

    /*
     * x: mass of the entity
     */
    ENTITY_MASS                 (cl_float),

    /*
     * x: model id of the aligned entity
     */
    ENTITY_MODEL_ID             (cl_int),

    /*
     * x: root hull index of the aligned entity
     */
    ENTITY_ROOT_HULL            (cl_int),

    /*
     * x: model transform index of the aligned entity
     */
    ENTITY_TRANSFORM_ID         (cl_int),


    // entity bones

    /*
     * s0-sF: Column-major, 4x4 transformation matrix, armature bone instance
     */
    ENTITY_BONE                 (cl_float16),

    /*
     * x: bind pose reference id
     */
    ENTITY_BONE_REFERENCE_ID    (cl_int),

    /*
     * x: armature bone parent id
     */
    ENTITY_BONE_PARENT_ID       (cl_int),

    //#endregion

    //#region Edge Buffers

    /*
     * x: point 1 index
     * y: point 2 index
     */
    EDGE                        (cl_int2),

    /*
     * x: edge flags (bit-field)
     */
    EDGE_FLAG                   (cl_int),

    /*
     * x: edge constraint length
     */
    EDGE_LENGTH                 (cl_float),

    /*
     * x: corner x position
     * y: corner y position
     * z: width
     * w: height
     */
    EDGE_AABB                   (cl_float4),

    /*
     * x: minimum x key index
     * y: maximum x key index
     * z: minimum y key index
     * w: maximum y key index
     */
    EDGE_AABB_INDEX             (cl_int4),

    /*
     * x: key bank offset
     * y: key bank size
     */
    EDGE_AABB_KEY_TABLE         (cl_int2),

    //#endregion

    //#region Hull Buffers

    /*
     * x: current x position
     * y: current y position
     * z: previous x position
     * w: previous y position
     */
    HULL                        (cl_float4),

    /*
     * x: scale x
     * y: scale y
     */
    HULL_SCALE                  (cl_float2),

    /*
     * x: corner x position
     * y: corner y position
     * z: width
     * w: height
     */
    HULL_AABB                   (cl_float4),

    /*
     * x: minimum x key index
     * y: maximum x key index
     * z: minimum y key index
     * w: maximum y key index
     */
    HULL_AABB_INDEX             (cl_int4),

    /*
     * x: key bank offset
     * y: key bank size
     */
    HULL_AABB_KEY_TABLE         (cl_int2),

    /*
     * x: entity id for aligned hull
     */
    HULL_ENTITY_ID              (cl_int),

    /*
     * x: start bone
     * y: end bone
     */
    HULL_BONE_TABLE             (cl_int2),

    /*
     * x: start point index
     * y: end point index
     */
    HULL_POINT_TABLE            (cl_int2),

    /*
     * x: start edge index
     * y: end edge index
     */
    HULL_EDGE_TABLE             (cl_int2),

    /*
     * x: hull flags (bit-field)
     */
    HULL_FLAG                   (cl_int),

    /*
     * x: friction coefficient
     */
    HULL_FRICTION               (cl_float),

    /*
     * x: restitution coefficient
     */
    HULL_RESTITUTION            (cl_float),

    /*
     * x: the integrity (i.e. health) of the hull
     */
    HULL_INTEGRITY              (cl_int),

    /*
     * x: reference mesh id
     */
    HULL_MESH_ID                (cl_int),

    /*
     * x: offset index of the UV to use for this hull
     */
    HULL_UV_OFFSET              (cl_int),

    /*
     * x: initial reference angle
     * y: current rotation
     */
    HULL_ROTATION               (cl_float2),

    // hull bones

    /*
     * s0-sF: Column-major, 4x4 transformation matrix, hull bone instance
     */
    HULL_BONE                   (cl_float16),

    /*
     * x: bone bind pose index (model space)
     */
    HULL_BONE_BIND_POSE         (cl_int),

    /*
     * x: bone inverse bind pose index (mesh-space)
     */
    HULL_BONE_INV_BIND_POSE     (cl_int),

    //#endregion

    //#region Point Buffers

    /*
     * x: current x position
     * y: current y position
     * z: previous x position
     * w: previous y position
     */
    POINT                       (cl_float4),

    /*
     * x: anti-gravity magnitude for each point
     */
    POINT_ANTI_GRAV             (cl_float),

    /*
     * x: anti-time magnitude for each point
     */
    POINT_ANTI_TIME             (cl_float),

    /*
     * x: bone 1 instance id
     * y: bone 2 instance id
     * z: bone 3 instance id
     * w: bone 4 instance id
     */
    POINT_BONE_TABLE            (cl_int4),

    /*
     * x: vertex flags (bit field)
     */
    POINT_FLAG                  (cl_int),

    /*
     * x: recent collision hit counter
     */
    POINT_HIT_COUNT             (cl_short),

    /*
     * x: hull index
     */
    POINT_HULL_INDEX            (cl_int),

    /*
     * x: reference vertex index
     */
    POINT_VERTEX_REFERENCE      (cl_int),

    //#endregion

    ;

    private final CLType item_size;
    CoreBufferType(CLType itemSize) { item_size = itemSize; }
    @Override public CLType data_type() { return item_size; }
}
