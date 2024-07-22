__kernel void prepare_edges(__global float4 *points, 
                            __global int *point_hull_indices,
                            __global int *hull_flags,
                            __global int2 *edges,
                            __global int *edge_flags,
                            __global float4 *vertex_vbo,
                            __global float2 *flag_vbo,
                            int offset,
                            int max_edge)
{
    int gid = get_global_id(0);
    if (gid >= max_edge) return;
    int edge_id = gid + offset;
    
    int2 edge = edges[edge_id];
    int p1_index = edge.x;
    int p2_index = edge.y;
    int flags = edge_flags[edge_id];
    int hull_id = point_hull_indices[p1_index];
    int h_flags = hull_flags[hull_id];
    bool isInterior = (flags & E_INTERIOR) !=0;
    bool is_sensor = (flags & SENSOR_EDGE) !=0;
    bool sensor_hit = (h_flags & SENSOR_HIT) !=0;

    float alpha = isInterior 
        ? .3 
        : sensor_hit 
            ? 5 
            : 1;
    float2 alpha_vec = (float2)(alpha, alpha);
    
    float4 p1 = points[p1_index];
    float4 p2 = points[p2_index];
    
    float2 p1_v = p1.xy;
    float2 p2_v = p2.xy;
    
    float4 d = (float4)(p1_v, p2_v);
    
    vertex_vbo[gid] = d;
    flag_vbo[gid] = alpha_vec;
}
