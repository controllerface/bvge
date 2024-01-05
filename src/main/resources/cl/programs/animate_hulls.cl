__kernel void animate_hulls(__global float4 *points,
                            __global float4 *hulls,
                            __global int2 *hull_flags,
                            __global int2 *vertex_tables,
                            __global float4 *armatures,
                            __global float2 *vertex_references,
                            __global float16 *bones)
{
    // todo: this kernel shoudl change to pulling out 4 bones and using the weights in reference vertex, 
    //  modifiy the point's location. 
    int gid = get_global_id(0);
    float4 point = points[gid];
    int2 vertex_table = vertex_tables[gid];
    
    float2 reference_vertex = vertex_references[vertex_table.x];

    // todo: this will be 4 bones that are used to adjust the point
    float16 bone = bones[vertex_table.y];
    
    float4 hull = hulls[vertex_table.y];
    int2 hull_flag = hull_flags[vertex_table.y];
    float4 armature = armatures[hull_flag.y]; 
    bool no_bones = (hull_flag.x & NO_BONES) !=0;
    if (no_bones) return;

    float4 padded = (float4)(reference_vertex.x, reference_vertex.y, 0.0f, 1.0f);
    float4 after_bone = matrix_transform(bone, padded);
    float2 un_padded = after_bone.xy;
    un_padded.x *= hull.z;
    un_padded.y *= hull.w;
    un_padded += armature.xy;
    point.x = un_padded.x;
    point.y = un_padded.y;
    points[gid] = point;
}