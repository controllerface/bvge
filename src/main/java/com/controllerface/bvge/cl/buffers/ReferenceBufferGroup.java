package com.controllerface.bvge.cl.buffers;

import static com.controllerface.bvge.cl.buffers.ReferenceBufferType.*;

public class ReferenceBufferGroup extends BufferGroup<ReferenceBufferType>
{
    public ReferenceBufferGroup(String name, long ptr_queue)
    {
        super(ReferenceBufferType.class, name, ptr_queue, true);

        // bone animation channels

        init_buffer(ANIM_POS_CHANNEL);
        init_buffer(ANIM_ROT_CHANNEL);
        init_buffer(ANIM_SCL_CHANNEL);
        init_buffer(ANIM_TIMING_INDEX);

        // keyframes

        init_buffer(ANIM_KEY_FRAME);
        init_buffer(ANIM_FRAME_TIME);

        // animation timings

        init_buffer(ANIM_DURATION);
        init_buffer(ANIM_TICK_RATE);

        // bind poses

        init_buffer(BONE_BIND_POSE);
        init_buffer(BONE_LAYER);
        init_buffer(BONE_ANIM_CHANNEL_TABLE);

        // bone references

        init_buffer(BONE_REFERENCE);

        // mesh faces

        init_buffer(MESH_FACE);

        // mesh tables

        init_buffer(MESH_VERTEX_TABLE);
        init_buffer(MESH_FACE_TABLE);

        // model transforms

        init_buffer(MODEL_TRANSFORM);

        // vertex references

        init_buffer(VERTEX_REFERENCE);
        init_buffer(VERTEX_UV_TABLE);
        init_buffer(VERTEX_WEIGHT);

        // texture UVs

        init_buffer(VERTEX_TEXTURE_UV);
    }

    public void ensure_bone_channel(int capacity)
    {
        buffer(ANIM_POS_CHANNEL).ensure_capacity(capacity);
        buffer(ANIM_ROT_CHANNEL).ensure_capacity(capacity);
        buffer(ANIM_SCL_CHANNEL).ensure_capacity(capacity);
        buffer(ANIM_TIMING_INDEX).ensure_capacity(capacity);
    }

    public void ensure_keyframe(int capacity)
    {
        buffer(ANIM_KEY_FRAME).ensure_capacity(capacity);
        buffer(ANIM_FRAME_TIME).ensure_capacity(capacity);
    }

    public void ensure_animation_timings(int capacity)
    {
        buffer(ANIM_DURATION).ensure_capacity(capacity);
        buffer(ANIM_TICK_RATE).ensure_capacity(capacity);
    }

    public void ensure_bind_pose(int capacity)
    {
        buffer(BONE_BIND_POSE).ensure_capacity(capacity);
        buffer(BONE_LAYER).ensure_capacity(capacity);
        buffer(BONE_ANIM_CHANNEL_TABLE).ensure_capacity(capacity);
    }

    public void ensure_bone_reference(int capacity)
    {
        buffer(BONE_REFERENCE).ensure_capacity(capacity);
    }

    public void ensure_mesh_face(int capacity)
    {
        buffer(MESH_FACE).ensure_capacity(capacity);
    }

    public void ensure_mesh(int capacity)
    {
        buffer(MESH_VERTEX_TABLE).ensure_capacity(capacity);
        buffer(MESH_FACE_TABLE).ensure_capacity(capacity);
    }

    public void ensure_model_transform(int capacity)
    {
        buffer(MODEL_TRANSFORM).ensure_capacity(capacity);
    }

    public void ensure_vertex_reference(int capacity)
    {
        buffer(VERTEX_REFERENCE).ensure_capacity(capacity);
        buffer(VERTEX_UV_TABLE).ensure_capacity(capacity);
        buffer(VERTEX_WEIGHT).ensure_capacity(capacity);
    }

    public void ensure_vertex_texture_uv(int capacity)
    {
        buffer(VERTEX_TEXTURE_UV).ensure_capacity(capacity);
    }
}
