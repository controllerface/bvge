__kernel void animate_hulls(__global float4 *points,
                            __global float4 *hulls,
                            __global int2 *hull_flags,
                            __global int2 *vertex_tables,
                            __global float4 *armatures,
                            __global int *armature_flags,
                            __global float2 *vertex_references,
                            __global float16 *bones)
{
    int gid = get_global_id(0);
    float4 point = points[gid];
    int2 vertex_table = vertex_tables[gid];
    float2 reference_vertex = vertex_references[vertex_table.x];
    float16 bone = bones[vertex_table.y];
    float4 hull = hulls[vertex_table.y];
    int2 hull_flag = hull_flags[vertex_table.y];
    float4 armature = armatures[hull_flag.y]; // todo: no hard code
    bool no_bones = (hull_flag.x & 0x08) !=0;
    if (no_bones) return;

    //float4 root_hull = hulls[armature_flag];

    //printf("armature: gid:%d id:%d x:%f y:%f", gid, vertex_table.y, armature.x, armature.y);

    float4 padded = (float4)(reference_vertex.x, reference_vertex.y, 0.0f, 1.0f);
    float4 after_bone = matrix_transform(bone, padded);
    //printf("DEBUG GPU: id: %d x:%f y:%f\n", vertex_table.x, reference_vertex.x, reference_vertex.y);
    //if (bone.x != 0.0f || bone.y != 0.0f) printf("bone debug: x:%f y:%f", bone.x, bone.y);

    float2 un_padded = after_bone.xy;

    un_padded.x *= hull.z;
    un_padded.y *= hull.w;

    un_padded += armature.xy;
  
    point.x = un_padded.x;
    point.y = un_padded.y;

    points[gid] = point;
}