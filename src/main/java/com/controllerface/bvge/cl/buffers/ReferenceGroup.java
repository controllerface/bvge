package com.controllerface.bvge.cl.buffers;

import static com.controllerface.bvge.cl.CLSize.*;
import static com.controllerface.bvge.cl.buffers.BufferType.*;

public class ReferenceGroup extends BufferGroup
{
    public ReferenceGroup(String name, long ptr_queue)
    {
        super(name, ptr_queue);

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

        set_buffer(ANIM_FRAME_TIME,   cl_float);
        set_buffer(ANIM_KEY_FRAME,    cl_float4);
        set_buffer(ANIM_DURATION,     cl_float);
        set_buffer(ANIM_TICK_RATE,    cl_float);

        /* int
         * x: animation timing index
         */
        set_buffer(ANIM_TIMING_INDEX, cl_int);

        set_buffer(BONE_ANIM_TABLE,   cl_int2);
        set_buffer(BONE_BIND_POSE,    cl_float16);
        set_buffer(BONE_REFERENCE,    cl_float16);
        set_buffer(MESH_FACE,         cl_int4);
        set_buffer(MESH_VERTEX_TABLE, cl_int2);
        set_buffer(MESH_FACE_TABLE,   cl_int2);
        set_buffer(MODEL_TRANSFORM,   cl_float16);
        set_buffer(VERTEX_REFERENCE,  cl_float2);
        set_buffer(VERTEX_TEXTURE_UV, cl_float2);
        set_buffer(VERTEX_UV_TABLE,   cl_int2);
        set_buffer(VERTEX_WEIGHT,     cl_float4);
    }

    public void ensure_bone_channel(int capacity)
    {
        get_buffer(ANIM_POS_CHANNEL).ensure_capacity(capacity);
        get_buffer(ANIM_ROT_CHANNEL).ensure_capacity(capacity);
        get_buffer(ANIM_SCL_CHANNEL).ensure_capacity(capacity);
        get_buffer(ANIM_TIMING_INDEX).ensure_capacity(capacity);
    }
}
