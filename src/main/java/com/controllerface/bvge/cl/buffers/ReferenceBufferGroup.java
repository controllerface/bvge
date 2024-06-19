package com.controllerface.bvge.cl.buffers;

import static com.controllerface.bvge.cl.CLSize.*;
import static com.controllerface.bvge.cl.buffers.BufferType.*;

public class ReferenceBufferGroup extends BufferGroup
{
    public ReferenceBufferGroup(String name, long ptr_queue)
    {
        super(name, ptr_queue);

        // bone animation channels

        /* int2
         * x: position channel start index
         * y: position channel end index
         */
        set_buffer(ANIM_POS_CHANNEL,  cl_int2);

        /* int2
         * x: rotation channel start index
         * y: rotation channel end index
         */
        set_buffer(ANIM_ROT_CHANNEL,  cl_int2);

        /* int2
         * x: scaling channel start index
         * y: scaling channel end index
         */
        set_buffer(ANIM_SCL_CHANNEL,  cl_int2);

        /* int
         * x: animation timing index
         */
        set_buffer(ANIM_TIMING_INDEX, cl_int);


        // keyframes

        /* float4
         * x: vector/quaternion x
         * y: vector/quaternion y
         * z: vector/quaternion z
         * w: vector/quaternion w
         */
        set_buffer(ANIM_KEY_FRAME,    cl_float4);

        /* float
         * x: key frame timestamp
         */
        set_buffer(ANIM_FRAME_TIME,   cl_float);


        // animation timings

        /* float
         * x: animation duration
         */
        set_buffer(ANIM_DURATION,     cl_float);

        /* float
         * x: animation tick rate (FPS)
         */
        set_buffer(ANIM_TICK_RATE,    cl_float);

        // bind poses

        /* float16
         * s0-sF: Column-major, 4x4 transformation matrix, mesh-space bone reference (inverse bind pose)
         */
        set_buffer(BONE_BIND_POSE,    cl_float16);

        /* int2
         * x: bone channel start index
         * y: bone channel end index
         */
        set_buffer(BONE_ANIM_CHANNEL_TABLE,   cl_int2);


        // bone references

        /* float16
         * s0-sF: Column-major, 4x4 transformation matrix, model-space bone reference (bind pose)
         */
        set_buffer(BONE_REFERENCE,    cl_float16);


        // mesh faces

        /* int4
         * x: vertex 1 index
         * y: vertex 2 index
         * z: vertex 3 index
         * w: parent reference mesh ID
         */
        set_buffer(MESH_FACE,         cl_int4);


        // mesh tables

        /* int2
         * x: start vertex index
         * y: end vertex index
         */
        set_buffer(MESH_VERTEX_TABLE, cl_int2);

        /* int2
         * z: start face index
         * w: end face index
         */
        set_buffer(MESH_FACE_TABLE,   cl_int2);


        // model transforms

        /* float16
         * s0-sF: Column-major, 4x4 transformation matrix
         */
        set_buffer(MODEL_TRANSFORM,   cl_float16);


        // vertex references

        /* float2
         * x: x position
         * y: y position
         */
        set_buffer(VERTEX_REFERENCE,  cl_float2);

        /* int2
         * x: start UV index
         * y: end UV index
         */
        set_buffer(VERTEX_UV_TABLE,   cl_int2);

        /* float4
         * x: bone 1 weight
         * y: bone 2 weight
         * z: bone 3 weight
         * w: bone 4 weight
         */
        set_buffer(VERTEX_WEIGHT,     cl_float4);


        // texture UVs

        /* float2
         * x: u coordinate
         * y: v coordinate
         */
        set_buffer(VERTEX_TEXTURE_UV, cl_float2);
    }

    public void ensure_bone_channel(int capacity)
    {
        get_buffer(ANIM_POS_CHANNEL).ensure_capacity(capacity);
        get_buffer(ANIM_ROT_CHANNEL).ensure_capacity(capacity);
        get_buffer(ANIM_SCL_CHANNEL).ensure_capacity(capacity);
        get_buffer(ANIM_TIMING_INDEX).ensure_capacity(capacity);
    }

    public void ensure_keyframe(int capacity)
    {
        get_buffer(ANIM_KEY_FRAME).ensure_capacity(capacity);
        get_buffer(ANIM_FRAME_TIME).ensure_capacity(capacity);
    }

    public void ensure_animation_timings(int capacity)
    {
        get_buffer(ANIM_DURATION).ensure_capacity(capacity);
        get_buffer(ANIM_TICK_RATE).ensure_capacity(capacity);
    }

    public void ensure_bind_pose(int capacity)
    {
        get_buffer(BONE_BIND_POSE).ensure_capacity(capacity);
        get_buffer(BONE_ANIM_CHANNEL_TABLE).ensure_capacity(capacity);
    }

    public void ensure_bone_reference(int capacity)
    {
        get_buffer(BONE_REFERENCE).ensure_capacity(capacity);
    }

    public void ensure_mesh_face(int capacity)
    {
        get_buffer(MESH_FACE).ensure_capacity(capacity);
    }

    public void ensure_mesh(int capacity)
    {
        get_buffer(MESH_VERTEX_TABLE).ensure_capacity(capacity);
        get_buffer(MESH_FACE_TABLE).ensure_capacity(capacity);
    }

    public void ensure_model_transform(int capacity)
    {
        get_buffer(MODEL_TRANSFORM).ensure_capacity(capacity);
    }

    public void ensure_vertex_reference(int capacity)
    {
        get_buffer(VERTEX_REFERENCE).ensure_capacity(capacity);
        get_buffer(VERTEX_UV_TABLE).ensure_capacity(capacity);
        get_buffer(VERTEX_WEIGHT).ensure_capacity(capacity);
    }

    public void ensure_vertex_texture_uv(int capacity)
    {
        get_buffer(VERTEX_TEXTURE_UV).ensure_capacity(capacity);
    }
}
