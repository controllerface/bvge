__kernel void prepare_entities(__global float4 *points,
                                __global float2 *vertex_vbo,
                                int offset,
                                int max_entity)
{

    int gid = get_global_id(0);
    if (gid >= max_entity) return;
    int point_id = gid + offset;
    float4 point = points[point_id];
    float2 p = point.xy;
    vertex_vbo[gid] = p;
}
