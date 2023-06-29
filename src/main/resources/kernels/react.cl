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
    if (manifold[0] == -1)
    {
        return;
    }

    // vertex and edge objects
    float16 vo = bodies[(int)manifold[0]];
    float16 eo = bodies[(int)manifold[1]];
    float2 normal;
    normal.x = manifold[2];
    normal.y = manifold[3];

    float2 collision_vector = normal * manifold[4];
    float vertex_magnitude = .5f;
    float edge_magnitude = .5f;

    // vertex reaction is easy
    float2 v_reaction = collision_vector * vertex_magnitude;

    // now do edge reactions
    float2 e1 = points[(int)manifold[5]].xy; 
    float2 e2 = points[(int)manifold[6]].xy;
    float2 collision_vertex = points[(int)manifold[7]].xy;
    float edge_contact = edgeContact(e1, e2, collision_vertex, collision_vector);
    //printf("debug: e1 x: %f y: %f", e1.x, e1.y);
    //printf("debug: e2 x: %f y: %f", e2.x, e2.y);
    //printf("debug: v  x: %f y: %f", collision_vertex.x, collision_vertex.y);

    float edge_scale = 1.0f / (edge_contact * edge_contact + (1 - edge_contact) * (1 - edge_contact));
    float2 e1_reaction = collision_vector * ((1 - edge_contact) * edge_magnitude * edge_scale);
    float2 e2_reaction = collision_vector * (edge_contact * edge_magnitude * edge_scale);

    // todo: detemine if everything is needed, if not may fit into smaller data type
    float16 reaction;
    reaction[0]  = (float)manifold[0];  // vertex object index
    reaction[1]  = (float)manifold[1];  // edge object index
    reaction[2]  = (float)manifold[2];  // normal x
    reaction[3]  = (float)manifold[3];  // normal y
    reaction[4]  = (float)manifold[4];  // min distance
    reaction[5]  = (float)manifold[5];  // edge point A
    reaction[6]  = (float)manifold[6];  // edge point B
    reaction[7]  = (float)manifold[7];  // vertex point
    reaction[8]  = v_reaction.x;        // vertex object reaction x1
    reaction[9]  = v_reaction.y;        // vertex object reaction y1
    reaction[10] = e1_reaction.x;       // edge object reaction x1
    reaction[11] = e1_reaction.y;       // edge object reaction y1
    reaction[12] = e2_reaction.x;       // edge object reaction x2
    reaction[13] = e2_reaction.y;       // edge object reaction y2            
    reaction[14] = (float)0;            // [empty]          
    reaction[15] = (float)0;            // [empty]        
    
    reactions[gid] = reaction;
}