/**
Prepares a vbo for rendering of edges as lines. The vbo will contain only a subset of 
the vertices that make up an edge, the star of the subset is defined by the offset value.
 */
__kernel void prepare_points(__global float4 *points, 
                             __global float2 *vertex_vbo,
                             int offset)
{
    int gid = get_global_id(0);
    int point_id = gid + offset;
    float4 point = points[point_id];
    float2 p = point.xy;
    vertex_vbo[gid] = p;
}