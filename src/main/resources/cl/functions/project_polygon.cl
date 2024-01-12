/**
Projects the points of the given hull onto the given normal vector.
The return vector contains the min/max distances, and the index of
the point that have the mininum value.
 */
inline float3 project_polygon(__global const float4 *points, 
                              __global const int4 *vertex_tables, 
                              int4 hull, 
                              float2 normal)
{
    int start = hull.x;
    int end   = hull.y;
	int vert_count = end - start + 1;

    float3 result;
    result.x = (float)0; // min
    result.y = (float)0; // max
    result.z = (float)0; // index
    bool minYet = false;
    bool maxYet = false;
    for (int i = 0; i < vert_count; i++)
    {
        int n = start + i;
        int4 vertex_table = vertex_tables[n];
        
        bool x = (vertex_table.z & 0x01) !=0;
        if (x) continue;

        float2 v = points[n].xy;
        float proj = dot(v, normal);
        if (proj < result.x || !minYet)
        {
            result.x = proj;
            result.z = n;
            minYet = true;
        }
        if (proj > result.y || !maxYet)
        {
            result.y = proj;
            maxYet = true;
        }
    }
    return result;
}