package com.controllerface.bvge.ecs.systems.sectors;

import com.controllerface.bvge.cl.CLSize;
import com.controllerface.bvge.cl.buffers.BufferGroup;
import com.controllerface.bvge.cl.buffers.BufferType;

public class OrderedSectorGroup extends BufferGroup
{
    public OrderedSectorGroup(long ptr_queue, long entity_init, long hull_init, long edge_init, long point_init)
    {
        super(ptr_queue);

        set_buffer(BufferType.EDGE,                     new_buffer(CLSize.cl_int2,    edge_init));
        set_buffer(BufferType.EDGE_LENGTH,              new_buffer(CLSize.cl_float,   edge_init));
        set_buffer(BufferType.EDGE_FLAG,                new_buffer(CLSize.cl_int,     edge_init));

        set_buffer(BufferType.HULL_BONE,                new_buffer(CLSize.cl_float16, hull_init));
        set_buffer(BufferType.HULL_BONE_BIND_POSE,      new_buffer(CLSize.cl_int,     hull_init));
        set_buffer(BufferType.HULL_BONE_INV_BIND_POSE,  new_buffer(CLSize.cl_int,     hull_init));

        set_buffer(BufferType.ENTITY_BONE,              new_buffer(CLSize.cl_float16));
        set_buffer(BufferType.ENTITY_BONE_REFERENCE_ID, new_buffer(CLSize.cl_int));
        set_buffer(BufferType.ENTITY_BONE_PARENT_ID,    new_buffer(CLSize.cl_int));

        set_buffer(BufferType.POINT,                    new_buffer(CLSize.cl_float4, point_init));
        set_buffer(BufferType.POINT_VERTEX_REFERENCE,   new_buffer(CLSize.cl_int,    point_init));
        set_buffer(BufferType.POINT_HULL_INDEX,         new_buffer(CLSize.cl_int,    point_init));
        set_buffer(BufferType.POINT_BONE_TABLE,         new_buffer(CLSize.cl_int4,   point_init));
        set_buffer(BufferType.POINT_HIT_COUNT,          new_buffer(CLSize.cl_short,  point_init));
        set_buffer(BufferType.POINT_FLAG,               new_buffer(CLSize.cl_int,    point_init));

        set_buffer(BufferType.HULL,                     new_buffer(CLSize.cl_float4, hull_init));
        set_buffer(BufferType.HULL_SCALE,               new_buffer(CLSize.cl_float2, hull_init));
        set_buffer(BufferType.HULL_POINT_TABLE,         new_buffer(CLSize.cl_int2,   hull_init));
        set_buffer(BufferType.HULL_EDGE_TABLE,          new_buffer(CLSize.cl_int2,   hull_init));
        set_buffer(BufferType.HULL_FLAG,                new_buffer(CLSize.cl_int,    hull_init));
        set_buffer(BufferType.HULL_BONE_TABLE,          new_buffer(CLSize.cl_int2,   hull_init));
        set_buffer(BufferType.HULL_ENTITY_ID,           new_buffer(CLSize.cl_int,    hull_init));
        set_buffer(BufferType.HULL_FRICTION,            new_buffer(CLSize.cl_float,  hull_init));
        set_buffer(BufferType.HULL_RESTITUTION,         new_buffer(CLSize.cl_float,  hull_init));
        set_buffer(BufferType.HULL_MESH_ID,             new_buffer(CLSize.cl_int,    hull_init));
        set_buffer(BufferType.HULL_UV_OFFSET,           new_buffer(CLSize.cl_int,    hull_init));
        set_buffer(BufferType.HULL_ROTATION,            new_buffer(CLSize.cl_float2, hull_init));
        set_buffer(BufferType.HULL_INTEGRITY,           new_buffer(CLSize.cl_int,    hull_init));

        set_buffer(BufferType.ENTITY_ANIM_ELAPSED,      new_buffer(CLSize.cl_float2, entity_init));
        set_buffer(BufferType.ENTITY_MOTION_STATE,      new_buffer(CLSize.cl_short2, entity_init));
        set_buffer(BufferType.ENTITY_ANIM_INDEX,        new_buffer(CLSize.cl_int2,   entity_init));
        set_buffer(BufferType.ENTITY,                   new_buffer(CLSize.cl_float4, entity_init));
        set_buffer(BufferType.ENTITY_FLAG,              new_buffer(CLSize.cl_int,    entity_init));
        set_buffer(BufferType.ENTITY_ROOT_HULL,         new_buffer(CLSize.cl_int,    entity_init));
        set_buffer(BufferType.ENTITY_MODEL_ID,          new_buffer(CLSize.cl_int,    entity_init));
        set_buffer(BufferType.ENTITY_TRANSFORM_ID,      new_buffer(CLSize.cl_int,    entity_init));
        set_buffer(BufferType.ENTITY_HULL_TABLE,        new_buffer(CLSize.cl_int2,   entity_init));
        set_buffer(BufferType.ENTITY_BONE_TABLE,        new_buffer(CLSize.cl_int2,   entity_init));
        set_buffer(BufferType.ENTITY_MASS,              new_buffer(CLSize.cl_float,  entity_init));
    }
}
