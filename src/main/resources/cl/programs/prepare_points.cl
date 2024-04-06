__kernel void prepare_points(__global float4 *points, 
                             __global float *anti_gravity,
                             __global float2 *vertex_vbo,
                             __global float4 *color_vbo,
                             int offset)
{
    int gid = get_global_id(0);
    int point_id = gid + offset;
    float4 point = points[point_id];
    float ag = anti_gravity[point_id];
    float2 p = point.xy;
    float col = ag > 0.0f ? 0.0f : 1.1f;
    float4 c = (float4)(1.0f, col, col, 1.0f);
    vertex_vbo[gid] = p;
    color_vbo[gid] = c;
}
