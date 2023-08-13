__kernel void animate_hulls(__global float4 *points,
                            __global float4 *hulls,
                            __global int2 *hull_flags,
                            __global int2 *vertex_tables,
                            __global float2 *armatures,
                            __global float2 *vertex_references,
                            __global float16 *bones)
{
    int gid = get_global_id(0);
    float4 point = points[gid];
    int2 vertex_table = vertex_tables[gid];
    float2 reference_vertex = vertex_references[vertex_table.x];
    float16 bone = bones[vertex_table.y];
    float4 hull = hulls[vertex_table.y];
    float2 armature = armatures[1]; // todo: no hard code
    int2 flags = hull_flags[vertex_table.y];
    bool no_bones = (flags.x & 0x08) !=0;
    if (no_bones) return;

    //printf("armature: gid:%d id:%d x:%f y:%f", gid, vertex_table.y, armature.x, armature.y);

    float4 padded = (float4)(reference_vertex.x, reference_vertex.y, 0.0f, 1.0f);
    float4 after_bone = matrix_transform(bone, padded);
    //printf("DEBUG GPU: id: %d x:%f y:%f\n", vertex_table.x, reference_vertex.x, reference_vertex.y);
    //if (bone.x != 0.0f || bone.y != 0.0f) printf("bone debug: x:%f y:%f", bone.x, bone.y);

    float2 un_padded = after_bone.xy;
    
    un_padded.x *= hull.z; // reuse z for uniform scale
    un_padded.y *= hull.z;

    un_padded.x += armature.x;
    un_padded.y += armature.y;
    
    // un_padded.x += point.x;
    // un_padded.y += point.y;
    // un_padded.x += hull.x;
    // un_padded.y += hull.y;
    // point.z = point.x;
    // point.w = point.y;

    point.x = un_padded.x;
    point.y = un_padded.y;
    // point.z = un_padded.x;
    // point.w = un_padded.y;
    points[gid] = point;
}