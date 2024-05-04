__kernel void prepare_liquids(__global float2 *hull_positions, 
                              __global float2 *hull_scales, 
                              __global float2 *hull_rotations,
                              __global int2 *hull_point_tables,
                              __global ushort *point_hit_counts,
                              __global int *indices,
                              __global float4 *transforms_out,
                              __global float4 *colors_out,
                              int offset)
{
    int gid = get_global_id(0);
    int offset_gid = gid + offset;
    int current_hull = indices[offset_gid];
    
    float2 position = hull_positions[current_hull];
    float2 scale    = hull_scales[current_hull];
    float2 rotation = hull_rotations[current_hull];
    int2 point_table = hull_point_tables[current_hull];

    int hit_counts = point_hit_counts[point_table.x];

    float4 col = hit_counts <= HIT_LOW_THRESHOLD
        ? (float4)(1.05f, 1.05f, 1.05f, 0.5f) 
        : hit_counts <= HIT_LOW_MID_THRESHOLD
            ? (float4)(1.0f) 
            : hit_counts <= HIT_MID_THRESHOLD
                ? (float4)(0.85f, 0.85f, 0.85f, 1.0f)
                : hit_counts <= HIT_HIGH_MID_THRESHOLD
                    ? (float4)(0.8f, 0.8f, 0.9f, 1.5f)
                    : (float4)(0.7f, 0.7f, 0.7f, 1.75f);

    float4 transform_out;
    transform_out.x = position.x; 
    transform_out.y = position.y; 
    transform_out.z = rotation.x;
    transform_out.w = scale.x; // note: uniform scale only

    transforms_out[gid] = transform_out;
    colors_out[gid] = col;
}
