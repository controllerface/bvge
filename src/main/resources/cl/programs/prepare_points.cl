__kernel void prepare_points(__global float4 *points, 
                             __global float *anti_gravity,
                             __global float2 *vertex_vbo,
                             __global float4 *color_vbo,
                             int offset,
                             int max_point)
{
    int gid = get_global_id(0);
    if (gid >= max_point) return;
    int point_id = gid + offset;
    float4 point = points[point_id];
    float ag = anti_gravity[point_id];
    float2 p = point.xy;
    float col = ag > 0.0f ? 0.0f : 0.5f;
    float4 c = (float4)(0.5f, col, col, 1.0f);
    vertex_vbo[gid] = p;
    color_vbo[gid] = c;
}
