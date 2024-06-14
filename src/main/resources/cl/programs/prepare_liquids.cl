inline float map(float x, float in_min, float in_max, float out_min, float out_max)
{
  return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min;
}

__kernel void prepare_liquids(__global float4 *hull_positions, 
                              __global float2 *hull_scales, 
                              __global float2 *hull_rotations,
                              __global int2 *hull_point_tables,
                              __global int *hull_uv_offsets,
                              __global short *point_hit_counts,
                              __global int *indices,
                              __global float4 *transforms_out,
                              __global float4 *colors_out,
                              int offset)
{
    int gid = get_global_id(0);
    int offset_gid = gid + offset;
    int current_hull = indices[offset_gid];
    
    float4 position  = hull_positions[current_hull];
    float2 scale     = hull_scales[current_hull];
    float2 rotation  = hull_rotations[current_hull];
    int2 point_table = hull_point_tables[current_hull];
    int uv_offset    = hull_uv_offsets[current_hull];

    int hit_counts = point_hit_counts[point_table.x];

    float col = map((float) hit_counts, 0, HIT_TOP_THRESHOLD, -0.01f, 0.25f);
    col = 1.0f - col;


    // float4 col = hit_counts <= HIT_LOW_THRESHOLD
    //     ? (float4)(1.0f, 1.0f, 1.0f, 0.5f) 
    //     : hit_counts <= HIT_LOW_MID_THRESHOLD
    //         ? (float4)(0.98f) 
    //         : hit_counts <= HIT_MID_THRESHOLD
    //             ? (float4)(0.96f, 0.96f, 0.96f, 1.0f)
    //             : hit_counts <= HIT_HIGH_MID_THRESHOLD
    //                 ? (float4)(0.94f, 0.94f, 0.94f, 1.5f)
    //                 : (float4)(0.92f, 0.92f, 0.92f, 2.0f);

    // float4 col = hit_counts <= HIT_LOW_THRESHOLD
    //     ? (float4)(1.0f, 1.0f, 1.0f, 0.5f) 
    //     : (float4)(0.8f, 0.8f, 0.8f, 2.0f);

    //float4 col = (float4)(1.0f, 1.0f, 1.0f, 2.0f);

    float4 c2 = liquid_lookup_table[uv_offset];

    c2 *= col;


    float4 transform_out;
    transform_out.x = position.x; 
    transform_out.y = position.y; 
    transform_out.z = rotation.x;
    transform_out.w = scale.x; // note: uniform scale only

    transforms_out[gid] = transform_out;
    colors_out[gid] = c2;
}
