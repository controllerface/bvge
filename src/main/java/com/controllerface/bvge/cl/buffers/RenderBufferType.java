package com.controllerface.bvge.cl.buffers;

import static com.controllerface.bvge.cl.CLSize.*;

public enum RenderBufferType implements BufferType
{
    RENDER_ENTITY                 (cl_float4),
    RENDER_ENTITY_FLAG            (cl_int),
    RENDER_ENTITY_MODEL_ID        (cl_int),
    RENDER_ENTITY_ROOT_HULL       (cl_int),
    RENDER_EDGE                   (cl_int2),
    RENDER_EDGE_FLAG              (cl_int),
    RENDER_EDGE_AABB              (cl_int4),
    RENDER_HULL                   (cl_float4),
    RENDER_HULL_AABB              (cl_int4),
    RENDER_HULL_ENTITY_ID         (cl_int),
    RENDER_HULL_FLAG              (cl_int),
    RENDER_HULL_MESH_ID           (cl_int),
    RENDER_HULL_UV_OFFSET         (cl_int),
    RENDER_HULL_INTEGRITY         (cl_int),
    RENDER_HULL_POINT_TABLE       (cl_int2),
    RENDER_HULL_ROTATION          (cl_float2),
    RENDER_HULL_SCALE             (cl_float2),
    RENDER_POINT                  (cl_float4),
    RENDER_POINT_ANTI_GRAV        (cl_float),
    RENDER_POINT_HIT_COUNT        (cl_short),
    RENDER_POINT_VERTEX_REFERENCE (cl_int),

    ;

    private final int item_size;
    RenderBufferType(int itemSize) { item_size = itemSize; }
    @Override public int size() { return item_size; }
}
