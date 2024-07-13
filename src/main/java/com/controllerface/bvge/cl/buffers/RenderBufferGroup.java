package com.controllerface.bvge.cl.buffers;

import static com.controllerface.bvge.cl.buffers.CoreBufferType.*;
import static com.controllerface.bvge.cl.buffers.RenderBufferType.*;

/**
 * Render "mirror" buffers are configured only for certain core buffers, and are used solely for rendering purposes.
 * Between physics simulation ticks, rendering threads use the mirror buffers to render the state of the objects
 * while the physics thread is busy calculating the data for the next frame.
 */
public class RenderBufferGroup extends BufferGroup<RenderBufferType>
{
    public RenderBufferGroup(String name, long ptr_queue, long entity_init, long hull_init, long edge_init, long point_init)
    {
        super(RenderBufferType.class, name, ptr_queue, true);

        set_buffer(RENDER_ENTITY,                 entity_init);
        set_buffer(RENDER_ENTITY_FLAG,            entity_init);
        set_buffer(RENDER_ENTITY_MODEL_ID,        entity_init);
        set_buffer(RENDER_ENTITY_ROOT_HULL,       entity_init);
        set_buffer(RENDER_EDGE,                   edge_init);
        set_buffer(RENDER_EDGE_FLAG,              edge_init);
        set_buffer(RENDER_EDGE_AABB,              point_init);
        set_buffer(RENDER_HULL,                   hull_init);
        set_buffer(RENDER_HULL_AABB,              hull_init);
        set_buffer(RENDER_HULL_ENTITY_ID,         hull_init);
        set_buffer(RENDER_HULL_FLAG,              hull_init);
        set_buffer(RENDER_HULL_MESH_ID,           hull_init);
        set_buffer(RENDER_HULL_UV_OFFSET,         hull_init);
        set_buffer(RENDER_HULL_INTEGRITY,         hull_init);
        set_buffer(RENDER_HULL_POINT_TABLE,       hull_init);
        set_buffer(RENDER_HULL_ROTATION,          hull_init);
        set_buffer(RENDER_HULL_SCALE,             hull_init);
        set_buffer(RENDER_POINT,                  point_init);
        set_buffer(RENDER_POINT_ANTI_GRAV,        point_init);
        set_buffer(RENDER_POINT_HIT_COUNT,        point_init);
        set_buffer(RENDER_POINT_VERTEX_REFERENCE, point_init);
    }

    public void copy_from(BufferGroup<CoreBufferType> sector_group)
    {
        get_buffer(RENDER_ENTITY).copy_from(sector_group.get_buffer(ENTITY));
        get_buffer(RENDER_ENTITY_FLAG).copy_from(sector_group.get_buffer(ENTITY_FLAG));
        get_buffer(RENDER_ENTITY_MODEL_ID).copy_from(sector_group.get_buffer(ENTITY_MODEL_ID));
        get_buffer(RENDER_ENTITY_ROOT_HULL).copy_from(sector_group.get_buffer(ENTITY_ROOT_HULL));
        get_buffer(RENDER_EDGE).copy_from(sector_group.get_buffer(EDGE));
        get_buffer(RENDER_EDGE_FLAG).copy_from(sector_group.get_buffer(EDGE_FLAG));
        get_buffer(RENDER_EDGE_AABB).copy_from(sector_group.get_buffer(EDGE_AABB));
        get_buffer(RENDER_HULL).copy_from(sector_group.get_buffer(HULL));
        get_buffer(RENDER_HULL_AABB).copy_from(sector_group.get_buffer(HULL_AABB));
        get_buffer(RENDER_HULL_FLAG).copy_from(sector_group.get_buffer(HULL_FLAG));
        get_buffer(RENDER_HULL_ENTITY_ID).copy_from(sector_group.get_buffer(HULL_ENTITY_ID));
        get_buffer(RENDER_HULL_MESH_ID).copy_from(sector_group.get_buffer(HULL_MESH_ID));
        get_buffer(RENDER_HULL_UV_OFFSET).copy_from(sector_group.get_buffer(HULL_UV_OFFSET));
        get_buffer(RENDER_HULL_INTEGRITY).copy_from(sector_group.get_buffer(HULL_INTEGRITY));
        get_buffer(RENDER_HULL_POINT_TABLE).copy_from(sector_group.get_buffer(HULL_POINT_TABLE));
        get_buffer(RENDER_HULL_ROTATION).copy_from(sector_group.get_buffer(HULL_ROTATION));
        get_buffer(RENDER_HULL_SCALE).copy_from(sector_group.get_buffer(HULL_SCALE));
        get_buffer(RENDER_POINT).copy_from(sector_group.get_buffer(POINT));
        get_buffer(RENDER_POINT_HIT_COUNT).copy_from(sector_group.get_buffer(POINT_HIT_COUNT));
        get_buffer(RENDER_POINT_ANTI_GRAV).copy_from(sector_group.get_buffer(POINT_ANTI_GRAV));
        get_buffer(RENDER_POINT_VERTEX_REFERENCE).copy_from(sector_group.get_buffer(POINT_VERTEX_REFERENCE));
    }
}
