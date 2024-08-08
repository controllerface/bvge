package com.controllerface.bvge.memory.groups;

import com.controllerface.bvge.gpu.cl.buffers.BufferGroup;
import com.controllerface.bvge.memory.types.CoreBufferType;
import com.controllerface.bvge.memory.types.RenderBufferType;

import static com.controllerface.bvge.memory.types.CoreBufferType.*;
import static com.controllerface.bvge.memory.types.RenderBufferType.*;

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

        init_buffer(RENDER_ENTITY,                 entity_init);
        init_buffer(RENDER_ENTITY_FLAG,            entity_init);
        init_buffer(RENDER_ENTITY_MODEL_ID,        entity_init);
        init_buffer(RENDER_ENTITY_ROOT_HULL,       entity_init);
        init_buffer(RENDER_EDGE,                   edge_init);
        init_buffer(RENDER_EDGE_FLAG,              edge_init);
        init_buffer(RENDER_HULL,                   hull_init);
        init_buffer(RENDER_HULL_AABB,              hull_init);
        init_buffer(RENDER_HULL_ENTITY_ID,         hull_init);
        init_buffer(RENDER_HULL_FLAG,              hull_init);
        init_buffer(RENDER_HULL_MESH_ID,           hull_init);
        init_buffer(RENDER_HULL_UV_OFFSET,         hull_init);
        init_buffer(RENDER_HULL_INTEGRITY,         hull_init);
        init_buffer(RENDER_HULL_POINT_TABLE,       hull_init);
        init_buffer(RENDER_HULL_ROTATION,          hull_init);
        init_buffer(RENDER_HULL_SCALE,             hull_init);
        init_buffer(RENDER_POINT,                  point_init);
        init_buffer(RENDER_POINT_ANTI_GRAV,        point_init);
        init_buffer(RENDER_POINT_HIT_COUNT,        point_init);
        init_buffer(RENDER_POINT_VERTEX_REFERENCE, point_init);
        init_buffer(RENDER_POINT_HULL_INDEX,       point_init);
    }

    public void copy_from(BufferGroup<CoreBufferType> sector_group)
    {
        buffer(RENDER_ENTITY).copy_from(sector_group.buffer(ENTITY));
        buffer(RENDER_ENTITY_FLAG).copy_from(sector_group.buffer(ENTITY_FLAG));
        buffer(RENDER_ENTITY_MODEL_ID).copy_from(sector_group.buffer(ENTITY_MODEL_ID));
        buffer(RENDER_ENTITY_ROOT_HULL).copy_from(sector_group.buffer(ENTITY_ROOT_HULL));
        buffer(RENDER_EDGE).copy_from(sector_group.buffer(EDGE));
        buffer(RENDER_EDGE_FLAG).copy_from(sector_group.buffer(EDGE_FLAG));
        buffer(RENDER_HULL).copy_from(sector_group.buffer(HULL));
        buffer(RENDER_HULL_AABB).copy_from(sector_group.buffer(HULL_AABB));
        buffer(RENDER_HULL_FLAG).copy_from(sector_group.buffer(HULL_FLAG));
        buffer(RENDER_HULL_ENTITY_ID).copy_from(sector_group.buffer(HULL_ENTITY_ID));
        buffer(RENDER_HULL_MESH_ID).copy_from(sector_group.buffer(HULL_MESH_ID));
        buffer(RENDER_HULL_UV_OFFSET).copy_from(sector_group.buffer(HULL_UV_OFFSET));
        buffer(RENDER_HULL_INTEGRITY).copy_from(sector_group.buffer(HULL_INTEGRITY));
        buffer(RENDER_HULL_POINT_TABLE).copy_from(sector_group.buffer(HULL_POINT_TABLE));
        buffer(RENDER_HULL_ROTATION).copy_from(sector_group.buffer(HULL_ROTATION));
        buffer(RENDER_HULL_SCALE).copy_from(sector_group.buffer(HULL_SCALE));
        buffer(RENDER_POINT).copy_from(sector_group.buffer(POINT));
        buffer(RENDER_POINT_HIT_COUNT).copy_from(sector_group.buffer(POINT_HIT_COUNT));
        buffer(RENDER_POINT_ANTI_GRAV).copy_from(sector_group.buffer(POINT_ANTI_GRAV));
        buffer(RENDER_POINT_VERTEX_REFERENCE).copy_from(sector_group.buffer(POINT_VERTEX_REFERENCE));
        buffer(RENDER_POINT_HULL_INDEX).copy_from(sector_group.buffer(POINT_HULL_INDEX));
    }
}
