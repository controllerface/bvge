inline float polygon_distance(float3 proj_a, float3 proj_b)
{
    if (proj_a.x < proj_b.x)
    {
        return proj_b.x - proj_a.y;
    }
    else
    {
        return proj_a.x - proj_b.y;
    }
}