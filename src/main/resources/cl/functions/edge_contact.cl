inline float edge_contact(float2 e1, float2 e2, float2 collision_vertex, float2 collision_vector)
{
    float contact;
    float x_dist = e1.x - e2.x;
    float y_dist = e1.y - e2.y;
    if (fabs(x_dist) > fabs(y_dist))
    {
        float x_offset = (collision_vertex.x - collision_vector.x - e1.x);
        float x_diff = (e2.x - e1.x);
        contact = native_divide(x_offset, x_diff);
    }
    else
    {
        float y_offset = (collision_vertex.y - collision_vector.y - e1.y);
        float y_diff = (e2.y - e1.y);
        contact = native_divide(y_offset, y_diff);
    }
    return contact;
}