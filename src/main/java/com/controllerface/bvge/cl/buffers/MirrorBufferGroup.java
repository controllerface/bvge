package com.controllerface.bvge.cl.buffers;

import static com.controllerface.bvge.cl.CLSize.*;
import static com.controllerface.bvge.cl.CLSize.cl_int;
import static com.controllerface.bvge.cl.buffers.CoreBufferType.*;
import static com.controllerface.bvge.cl.buffers.MirrorBufferType.*;

/**
 * Mirror buffers are configured only for certain core buffers, and are used solely for rendering purposes.
 * Between physics simulation ticks, rendering threads use the mirror buffers to render the state of the objects
 * while the physics thread is busy calculating the data for the next frame.
 */
public class MirrorBufferGroup extends BufferGroup<MirrorBufferType>
{
    public MirrorBufferGroup(String name, long ptr_queue, long entity_init, long hull_init, long edge_init, long point_init)
    {
        super(MirrorBufferType.class, name, ptr_queue, true);

        set_buffer(MIRROR_ENTITY,                 cl_float4, entity_init);
        set_buffer(MIRROR_ENTITY_FLAG,            cl_int,    entity_init);
        set_buffer(MIRROR_ENTITY_MODEL_ID,        cl_int,    entity_init);
        set_buffer(MIRROR_ENTITY_ROOT_HULL,       cl_int,    entity_init);
        set_buffer(MIRROR_EDGE,                   cl_int2,   edge_init);
        set_buffer(MIRROR_EDGE_FLAG,              cl_int,    edge_init);
        set_buffer(MIRROR_HULL,                   cl_float4, hull_init);
        set_buffer(MIRROR_HULL_AABB,              cl_int4,   hull_init);
        set_buffer(MIRROR_HULL_FLAG,              cl_int,    hull_init);
        set_buffer(MIRROR_HULL_ENTITY_ID,         cl_int,    hull_init);
        set_buffer(MIRROR_HULL_MESH_ID,           cl_int,    hull_init);
        set_buffer(MIRROR_HULL_UV_OFFSET,         cl_int,    hull_init);
        set_buffer(MIRROR_HULL_INTEGRITY,         cl_int,    hull_init);
        set_buffer(MIRROR_HULL_POINT_TABLE,       cl_int2,   hull_init);
        set_buffer(MIRROR_HULL_ROTATION,          cl_float2, hull_init);
        set_buffer(MIRROR_HULL_SCALE,             cl_float2, hull_init);
        set_buffer(MIRROR_POINT,                  cl_float4, point_init);
        set_buffer(MIRROR_POINT_AABB,             cl_int4,   point_init);
        set_buffer(MIRROR_POINT_HIT_COUNT,        cl_short,  point_init);
        set_buffer(MIRROR_POINT_ANTI_GRAV,        cl_float,  point_init);
        set_buffer(MIRROR_POINT_VERTEX_REFERENCE, cl_int,    point_init);
    }

    public void mirror(BufferGroup<CoreBufferType> sector_group)
    {
        get_buffer(MIRROR_ENTITY).mirror(sector_group.get_buffer(ENTITY));
        get_buffer(MIRROR_ENTITY_FLAG).mirror(sector_group.get_buffer(ENTITY_FLAG));
        get_buffer(MIRROR_ENTITY_MODEL_ID).mirror(sector_group.get_buffer(ENTITY_MODEL_ID));
        get_buffer(MIRROR_ENTITY_ROOT_HULL).mirror(sector_group.get_buffer(ENTITY_ROOT_HULL));
        get_buffer(MIRROR_EDGE).mirror(sector_group.get_buffer(EDGE));
        get_buffer(MIRROR_EDGE_FLAG).mirror(sector_group.get_buffer(EDGE_FLAG));
        get_buffer(MIRROR_HULL).mirror(sector_group.get_buffer(HULL));
        get_buffer(MIRROR_HULL_AABB).mirror(sector_group.get_buffer(HULL_AABB));
        get_buffer(MIRROR_HULL_FLAG).mirror(sector_group.get_buffer(HULL_FLAG));
        get_buffer(MIRROR_HULL_ENTITY_ID).mirror(sector_group.get_buffer(HULL_ENTITY_ID));
        get_buffer(MIRROR_HULL_MESH_ID).mirror(sector_group.get_buffer(HULL_MESH_ID));
        get_buffer(MIRROR_HULL_UV_OFFSET).mirror(sector_group.get_buffer(HULL_UV_OFFSET));
        get_buffer(MIRROR_HULL_INTEGRITY).mirror(sector_group.get_buffer(HULL_INTEGRITY));
        get_buffer(MIRROR_HULL_POINT_TABLE).mirror(sector_group.get_buffer(HULL_POINT_TABLE));
        get_buffer(MIRROR_HULL_ROTATION).mirror(sector_group.get_buffer(HULL_ROTATION));
        get_buffer(MIRROR_HULL_SCALE).mirror(sector_group.get_buffer(HULL_SCALE));
        get_buffer(MIRROR_POINT).mirror(sector_group.get_buffer(POINT));
        get_buffer(MIRROR_POINT_AABB).mirror(sector_group.get_buffer(POINT_AABB));
        get_buffer(MIRROR_POINT_HIT_COUNT).mirror(sector_group.get_buffer(POINT_HIT_COUNT));
        get_buffer(MIRROR_POINT_ANTI_GRAV).mirror(sector_group.get_buffer(POINT_ANTI_GRAV));
        get_buffer(MIRROR_POINT_VERTEX_REFERENCE).mirror(sector_group.get_buffer(POINT_VERTEX_REFERENCE));
    }
}
