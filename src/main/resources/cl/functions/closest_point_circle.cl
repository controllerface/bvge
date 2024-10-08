/**
Returns the index of the closest point on the polygon to the center of the circle
 */
inline int closest_point_circle(float2 circle_center, 
                                int2 polygon_point_table, 
                                __global const float4 *points,
                                __global int *point_flags)
{
    int result = -1;
    float minDistance = FLT_MAX;

    int start = polygon_point_table.x;
    int end   = polygon_point_table.y;
	int vert_count = end - start + 1;
    for (int i = 0; i < vert_count; i++)
    {
        int n = start + i;
        int point_flag = point_flags[n];
        
        bool is_interior = (point_flag & P_INTERIOR) !=0;
        if (is_interior) continue;

        float2 v = points[n].xy;
        float _distance = fast_distance(v, circle_center);
        if(_distance < minDistance)
        {
            minDistance = _distance;
            result = i;
        }
    }

    return result;
}
