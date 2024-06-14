/**
Handles collision between one polygonal hull and one circular hull
 */
inline bool point_polygon_containment(int polygon_id, 
                                     float2 test_point,
                                     __global int2 *hull_edge_tables,
                                     __global float4 *points,
                                     __global int2 *edges,
                                     __global int *edge_flags)
{
    int2 polygon_edge_table = hull_edge_tables[polygon_id];
	int polygon_edge_count = polygon_edge_table.y - polygon_edge_table.x + 1;

    float initial_sign = 0;
    for (int point_index = 0; point_index < polygon_edge_count; point_index++)
    {
        int edge_index = polygon_edge_table.x + point_index;
        int2 edge = edges[edge_index];
        int edge_flag = edge_flags[edge_index];
        
        // do not test interior edges
        if (edge_flag == 1) continue;

        int a_index = edge.x;
        int b_index = edge.y;

        float2 va = points[a_index].xy;
        float2 vb = points[b_index].xy;

        float c = (vb.x - va.x) * (test_point.y - va.y) - (vb.y - va.y) * (test_point.x - va.x);

        if (c != 0) 
        {
            if (initial_sign == 0) 
            {
                initial_sign = c;
            } 
            else 
            {
                if (initial_sign * c < 0) 
                {
                    return false; // point is outside
                }
            }
        }
    }
    return true; // point is inside
}
