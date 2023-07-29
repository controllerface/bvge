/**
This is a collection of Create/Read/Update/Delete (CRUD) functions that are used
to query and update objects stored on the GPU. Unlike most kernels, these functions
are designed to operate on a single target object. 
 */

 // todo: convert to float 4, transform

__kernel void read_position(__global float4 *bodies,
                            __global float *output,
                            int target)
{
    float4 body = bodies[target];
    output[0] = body.x;
    output[1] = body.y;
}

__kernel void update_accel(__global float2 *body_accel,
                           int target,
                           float2 new_value)
{
    float2 accel = body_accel[target];
    accel.x = new_value.x;
    accel.y = new_value.y;
    body_accel[target] = accel;
}

// todo: convert to float 4 transform,

__kernel void rotate_body(__global float4 *bodies,
                          __global int4 *element_tables,
                          __global float4 *points,
                          int target,
                          float angle)
{
    float4 body = bodies[target];
    int4 element_table = element_tables[target];
    int start = element_table.x;
    int end   = element_table.y;
    float2 origin = (float2)(body.x, body.y);
    for (int i = start; i <= end; i++)
    {
        float4 point = points[i];
        points[i] = rotate_point(point, origin, angle);
    }
}

__kernel void create_point(__global float4 *points,
                           int target,
                           float4 new_point)
{
    points[target] = new_point; 
}

__kernel void create_edge(__global float4 *edges,
                           int target,
                           float4 new_edge)
{
    edges[target] = new_edge; 
}

// todo: convert to float 4, transform

__kernel void create_body(__global float4 *bodies,
                          __global int4 *element_tables,
                          __global int *body_flags,
                          int target,
                          float4 new_body,
                          int4 new_table,
                          int new_flags)
{
    bodies[target] = new_body; 
    element_tables[target] = new_table; 
    body_flags[target] = new_flags; 
}


// new world below - Bulk methods

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