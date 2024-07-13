package com.controllerface.bvge.cl.buffers;

import static com.controllerface.bvge.cl.buffers.ReferenceBufferType.*;

public class ReferenceBufferGroup extends BufferGroup<ReferenceBufferType>
{
    public ReferenceBufferGroup(String name, long ptr_queue)
    {
        super(ReferenceBufferType.class, name, ptr_queue, true);

        // bone animation channels

        set_buffer(ANIM_POS_CHANNEL);
        set_buffer(ANIM_ROT_CHANNEL);
        set_buffer(ANIM_SCL_CHANNEL);
        set_buffer(ANIM_TIMING_INDEX);

        // keyframes

        set_buffer(ANIM_KEY_FRAME);
        set_buffer(ANIM_FRAME_TIME);

        // animation timings

        set_buffer(ANIM_DURATION);
        set_buffer(ANIM_TICK_RATE);

        // bind poses

        set_buffer(BONE_BIND_POSE);
        set_buffer(BONE_ANIM_CHANNEL_TABLE);

        // bone references

        set_buffer(BONE_REFERENCE);

        // mesh faces

        set_buffer(MESH_FACE);

        // mesh tables

        set_buffer(MESH_VERTEX_TABLE);
        set_buffer(MESH_FACE_TABLE);

        // model transforms

        set_buffer(MODEL_TRANSFORM);

        // vertex references

        set_buffer(VERTEX_REFERENCE);
        set_buffer(VERTEX_UV_TABLE);
        set_buffer(VERTEX_WEIGHT);

        // texture UVs

        set_buffer(VERTEX_TEXTURE_UV);
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
