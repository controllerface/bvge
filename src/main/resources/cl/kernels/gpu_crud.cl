/**
This is a collection of Create/Read/Update/Delete (CRUD) functions that are used
to query and update objects stored on the GPU. Unlike most kernels, these functions
are designed to operate on a single target object. 
 */
__kernel void read_position(__global float4 *hulls,
                            __global float *output,
                            int target)
{
    float4 hull = hulls[target];
    output[0] = hull.x;
    output[1] = hull.y;
}

__kernel void update_accel(__global float2 *hull_accel,
                           int target,
                           float2 new_value)
{
    float2 accel = hull_accel[target];
    accel.x = new_value.x;
    accel.y = new_value.y;
    hull_accel[target] = accel;
}

__kernel void rotate_hull(__global float4 *hulls,
                          __global int4 *element_tables,
                          __global float4 *points,
                          int target,
                          float angle)
{
    float4 hull = hulls[target];
    int4 element_table = element_tables[target];
    int start = element_table.x;
    int end   = element_table.y;
    float2 origin = (float2)(hull.x, hull.y);
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

__kernel void create_hull(__global float4 *hulls,
                          __global float2 *hull_rotations,
                          __global int4 *element_tables,
                          __global int2 *hull_flags,
                          int target,
                          float4 new_hull,
                          float2 new_rotation,
                          int4 new_table,
                          int2 new_flags)
{
    hulls[target] = new_hull; 
    hull_rotations[target] = new_rotation; 
    element_tables[target] = new_table; 
    hull_flags[target] = new_flags; 
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

__kernel void create_hulls(__global int *indices,
                            __global float4 *new_hulls,
                            __global int4 *new_tables,
                            __global float4 *hulls,
                            __global int4 *tables)
{
    int gid = get_global_id(0);
    int hull_index = indices[gid];
    float4 new_hull = new_hulls[gid];
    int4 new_table = new_tables[gid];
    hulls[hull_index] = new_hull; 
    tables[hull_index] = new_table; 
}