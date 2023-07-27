__kernel void read_position(__global float16 *bodies,
                            __global float *output,
                            int target)
{
    float16 body = bodies[target];
    output[0] = body.s0;
    output[1] = body.s1;
}

__kernel void update_accel(__global float16 *bodies,
                           int target,
                           float2 new_value)
{
    float16 body = bodies[target];
    body.s4 = new_value.x;
   	body.s5 = new_value.y;
    bodies[target] = body;
}

__kernel void rotate_body(__global float16 *bodies,
                          __global float4 *points,
                          int target,
                          float angle)
{
    float16 body = bodies[target];
    int start = (int)body.s7;
    int end   = (int)body.s8;
    float2 origin = (float2)(body.s0, body.s1);
    for (int i = start; i <= end; i++)
    {
        float4 point = points[i];
        points[i] = rotate_point(point, origin, angle);
    }
}



// new world below

__kernel void create_points(__global int *indices,
                            __global float4 *new_points,
                            __global float4 *points)
{
    int gid = get_global_id(0);
    int point_index = indices[gid];
    float4 new_point = new_points[gid];
    points[point_index] = new_point; 
}

__kernel void create_edges(__global int *indices,
                           __global float4 *new_edges,
                           __global float4 *edges)
{
    int gid = get_global_id(0);
    int edge_index = indices[gid];
    float4 new_edge = new_edges[gid];
    edges[edge_index] = new_edge; 
}

__kernel void create_bodies(__global int *indices,
                            __global float4 *new_bodies,
                            __global int4 *new_tables,
                            __global float4 *bodies,
                            __global int4 *tables)
{
    int gid = get_global_id(0);
    int body_index = indices[gid];
    float4 new_body = new_bodies[gid];
    int4 new_table = new_tables[gid];
    bodies[body_index] = new_body; 
    tables[body_index] = new_table; 
}