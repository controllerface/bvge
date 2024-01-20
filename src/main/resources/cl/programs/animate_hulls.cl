__kernel void animate_hulls(__global float4 *points,
                            __global float4 *hulls,
                            __global int4 *hull_flags,
                            __global int4 *vertex_tables,
                            __global int4 *bone_tables,
                            __global float4 *vertex_weights,
                            __global float4 *armatures,
                            __global float2 *vertex_references,
                            __global float16 *bones)
{

    int gid = get_global_id(0);
    float4 point = points[gid];
    int4 vertex_table = vertex_tables[gid];
    int4 hull_flag = hull_flags[vertex_table.y];
    bool no_bones = (hull_flag.x & NO_BONES) !=0;
    if (no_bones) return;

    int4 bone_table = bone_tables[gid];

    float2 reference_vertex = vertex_references[vertex_table.x];

    // todo: a separate kernel msut be called that actually updates all bones using
    // an animation. Right now, all bones are in bind pose and then never change.

    // todo: use all four bones with weights
    float16 bone = bones[bone_table.x];
    
    float4 hull = hulls[vertex_table.y];
    float4 armature = armatures[hull_flag.y]; 

    float4 padded = (float4)(reference_vertex.x, reference_vertex.y, 0.0f, 1.0f);
    float4 after_bone = matrix_transform(bone, padded);
    float2 un_padded = after_bone.xy;
    
    // this is effectively a model transform with just scale and position
    un_padded.x *= hull.z;
    un_padded.y *= hull.w;
    un_padded += armature.xy;
    point.x = un_padded.x;
    point.y = un_padded.y;
    points[gid] = point;
}