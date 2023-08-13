inline float4 transform(float16 matrix, float4 vector)
{
    float4 result;
    result.x = matrix.s0 * vector.x + matrix.s4 * vector.y + matrix.s8 * vector.z + matrix.sC * vector.w;
    result.y = matrix.s1 * vector.x + matrix.s5 * vector.y + matrix.s9 * vector.z + matrix.sD * vector.w;
    result.z = matrix.s2 * vector.x + matrix.s6 * vector.y + matrix.sA * vector.z + matrix.sE * vector.w;
    result.w = matrix.s3 * vector.x + matrix.s7 * vector.y + matrix.sB * vector.z + matrix.sF * vector.w;
    return result;
}

__kernel void animate_hulls(__global float4 *points,
                            __global float4 *hulls,
                            __global int2 *hull_flags,
                            __global int2 *vertex_tables,
                            __global float2 *vertex_references,
                            __global float16 *bones)
{
    int gid = get_global_id(0);
    float4 point = points[gid];
    int2 vertex_table = vertex_tables[gid];
    float2 reference_vertex = vertex_references[vertex_table.x];
    float16 bone = bones[vertex_table.y];
    float4 hull = hulls[vertex_table.y];
    int2 flags = hull_flags[vertex_table.y];
    bool is_circle = (flags.x & 0x02) !=0;
    if (is_circle) return;
    
    float4 padded = (float4)(reference_vertex.x, reference_vertex.y, 0.0f, 0.0f);
    float4 after_bone = transform(bone, padded);
    float2 un_padded = after_bone.xy;
    un_padded.x *= hull.z;
    un_padded.y *= hull.w;
    un_padded.x += point.x;
    un_padded.y += point.y;
    point.x = un_padded.x;
    point.y = un_padded.y;
    points[gid] = point;
}