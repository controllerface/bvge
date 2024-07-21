/**
Projects the points of the given hull onto the given normal vector.
The return vector contains the min/max distances, and the index of
the point that have the mininum value.
 */
inline float3 project_polygon(__global const float4 *points, 
                              __global int *point_flags, 
                              int2 hull_point_table, 
                              float2 normal)
{
    int start = hull_point_table.x;
    int end   = hull_point_table.y;
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
        int point_flag = point_flags[n];
        
        bool is_interior = (point_flag & P_INTERIOR) !=0;
        if (is_interior) continue;

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
