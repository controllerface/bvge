__kernel void prepare_entities(__global float4 *points,
                                __global float2 *vertex_vbo,
                                int offset)
{

    int gid = get_global_id(0);
    int point_id = gid + offset;
    float4 point = points[point_id];
    float2 p = point.xy;
    vertex_vbo[gid] = p;
}
