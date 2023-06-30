float edgeContact(float2 e1, float2 e2, float2 collision_vertex, float2 collision_vector)
{
    float contact;
    float x_dist = e1.x - e2.x;
    float y_dist = e1.y - e2.y;
    if (fabs(x_dist) > fabs(y_dist))
    {
        float x_offset = (collision_vertex.x - collision_vector.x - e1.x);
        float x_diff = (e2.x - e1.x);
        contact = x_offset / x_diff;
    }
    else
    {
        float y_offset = (collision_vertex.y - collision_vector.y - e1.y);
        float y_diff = (e2.y - e1.y);
        contact = y_offset / y_diff;
    }
    return contact;
}

__kernel void react(
    __global const float8 *manifolds,
    __global const float16 *bodies,
    __global const float4 *points,
    __global float16 *reactions)
{
    int gid = get_global_id(0);
    float8 manifold = manifolds[gid];
    float16 reaction;
    reaction.s0 = -1;
    if (manifold.s0 == -1)
    {
        reactions[gid] = reaction;
        return;
    }

    // vertex and edge objects
    float16 vo = bodies[(int)manifold.s0];
    float16 eo = bodies[(int)manifold.s1];
    float2 normal;
    normal.x = manifold.s2;
    normal.y = manifold.s3;

    float2 collision_vector = normal * manifold.s4;
    float vertex_magnitude = .5f;
    float edge_magnitude = .5f;

    // vertex reaction is easy
    float2 v_reaction = collision_vector * vertex_magnitude;

    // now do edge reactions
    float2 e1 = points[(int)manifold.s5].xy;
    float2 e2 = points[(int)manifold.s6].xy;
    float2 collision_vertex = points[(int)manifold.s7].xy;
    float edge_contact = edgeContact(e1, e2, collision_vertex, collision_vector);

    float edge_scale = 1.0f / (edge_contact * edge_contact + (1 - edge_contact) * (1 - edge_contact));
    float2 e1_reaction = collision_vector * ((1 - edge_contact) * edge_magnitude * edge_scale);
    float2 e2_reaction = collision_vector * (edge_contact * edge_magnitude * edge_scale);

    // todo: detemine if everything is needed, if not may fit into smaller data type
    reaction.s0  = (float)manifold.s0;  // vertex object index
    reaction.s1  = (float)manifold.s1;  // edge object index
    reaction.s2  = (float)manifold.s2;  // normal x
    reaction.s3  = (float)manifold.s3;  // normal y
    reaction.s4  = (float)manifold.s4;  // min distance
    reaction.s5  = (float)manifold.s5;  // edge point A
    reaction.s6  = (float)manifold.s6;  // edge point B
    reaction.s7  = (float)manifold.s7;  // vertex point
    reaction.s8  = v_reaction.x;        // vertex object reaction x1
    reaction.s9  = v_reaction.y;        // vertex object reaction y1
    reaction.sa = e1_reaction.x;       // edge object reaction x1
    reaction.sb = e1_reaction.y;       // edge object reaction y1
    reaction.sc = e2_reaction.x;       // edge object reaction x2
    reaction.sd = e2_reaction.y;       // edge object reaction y2
    reaction.se = (float)0;            // [empty]
    reaction.sf = (float)0;            // [empty]
    
    reactions[gid] = reaction;
}