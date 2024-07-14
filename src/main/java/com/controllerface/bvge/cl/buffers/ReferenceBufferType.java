package com.controllerface.bvge.cl.buffers;

import static com.controllerface.bvge.cl.CLData.*;

public enum ReferenceBufferType implements BufferType
{
    /*
     * x: position channel start index
     * y: position channel end index
     */
    ANIM_POS_CHANNEL        (cl_int2),

    /*
     * x: rotation channel start index
     * y: rotation channel end index
     */
    ANIM_ROT_CHANNEL        (cl_int2),

    /*
     * x: scaling channel start index
     * y: scaling channel end index
     */
    ANIM_SCL_CHANNEL        (cl_int2),

    /*
     * x: x position
     * y: y position
     */
    VERTEX_REFERENCE        (cl_float2),

    /*
     * x: start UV index
     * y: end UV index
     */
    VERTEX_UV_TABLE         (cl_int2),

    /*
     * x: bone 1 weight
     * y: bone 2 weight
     * z: bone 3 weight
     * w: bone 4 weight
     */
    VERTEX_WEIGHT           (cl_float4),

    /*
     * x: u coordinate
     * y: v coordinate
     */
    VERTEX_TEXTURE_UV       (cl_float2),

    /*
     * s0-sF: Column-major, 4x4 transformation matrix
     */
    MODEL_TRANSFORM         (cl_float16),

    /*
     * x: animation timing index
     */
    ANIM_TIMING_INDEX       (cl_int),

    /*
     * x: start vertex index
     * y: end vertex index
     */
    MESH_VERTEX_TABLE       (cl_int2),

    /*
     * z: start face index
     * w: end face index
     */
    MESH_FACE_TABLE         (cl_int2),

    /*
     * x: vertex 1 index
     * y: vertex 2 index
     * z: vertex 3 index
     * w: parent reference mesh ID
     */
    MESH_FACE               (cl_int4),

    /*
     * s0-sF: Column-major, 4x4 transformation matrix, model-space bone reference (bind pose)
     */
    BONE_REFERENCE          (cl_float16),

    /*
     * x: bone channel start index
     * y: bone channel end index
     */
    BONE_ANIM_CHANNEL_TABLE (cl_int2),

    /*
     * s0-sF: Column-major, 4x4 transformation matrix, mesh-space bone reference (inverse bind pose)
     */
    BONE_BIND_POSE          (cl_float16),

    /*
     * x: animation duration
     */
    ANIM_DURATION           (cl_float),

    /*
     * x: animation tick rate (FPS)
     */
    ANIM_TICK_RATE          (cl_float),

    /*
     * x: vector/quaternion x
     * y: vector/quaternion y
     * z: vector/quaternion z
     * w: vector/quaternion w
     */
    ANIM_KEY_FRAME          (cl_float4),

    /*
     * x: key frame timestamp
     */
    ANIM_FRAME_TIME         (cl_float),

    ;

    private final CLType item_size;
    ReferenceBufferType(CLType itemSize) { item_size = itemSize; }
    @Override public CLType data_type() { return item_size; }
}
