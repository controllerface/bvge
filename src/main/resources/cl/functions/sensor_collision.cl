
inline float calculateDisplacementToEdge(float2 point, float4 edge) 
{
    float2 a = edge.xy;
    float2 b = edge.zw;

    float dx = b.x - a.x;
    float dy = b.y - a.y;

    float edgeLengthSquared = dx * dx + dy * dy;
    float t = ((point.x - a.x) * dx + (point.y - a.y) * dy) / edgeLengthSquared;

    if (t < 0) 
    {
        t = 0;
    } 
    else if (t > 1) 
    {
        t = 1;
    }

    float nearestX = a.x + t * dx;
    float nearestY = a.y + t * dy;

    float distanceX = point.x - nearestX;
    float distanceY = point.y - nearestY;

    if (distanceY > 0) 
    {
        return distanceY;
    } 
    else
    {
        return FLT_MAX;
    }
}

inline float calculateDisplacement(int polygon_id, 
                                     float2 test_point,
                                     __global int2 *hull_edge_tables,
                                     __global float4 *points,
                                     __global int2 *edges,
                                     __global int *edge_flags)                                      
{
    float minDisplacement = FLT_MAX;
    
    int2 polygon_edge_table = hull_edge_tables[polygon_id];
	int polygon_edge_count = polygon_edge_table.y - polygon_edge_table.x + 1;

    float initial_sign = 0;
    for (int point_index = 0; point_index < polygon_edge_count; point_index++)
    {
        int edge_index = polygon_edge_table.x + point_index;
        int2 edge = edges[edge_index];
        int edge_flag = edge_flags[edge_index];
        
        // do not test interior edges
        if ((edge_flag && E_INTERIOR) != 0) continue;

        int a_index = edge.x;
        int b_index = edge.y;

        float2 va = points[a_index].xy;
        float2 vb = points[b_index].xy;

        float displacement = calculateDisplacementToEdge(test_point, (float4)(va, vb));
        if (displacement >= 0 && displacement < minDisplacement) 
        {
            minDisplacement = displacement;
        }
    }
    return minDisplacement == FLT_MAX ? 0 : minDisplacement;
}

/**
Handles collision between two polygonal hulls
 */
void polygon_sensor_collision(int polygon_id, 
                              int sensor_id,
                              __global float4 *hulls,
                              __global float4 *points,
                              __global int2 *edges,
                              __global int2 *hull_point_tables,
                              __global int2 *hull_edge_tables,
                              __global int *edge_flags,
                              __global int *hull_flags,
                              __global int *counter,
                              __global float8 *reactions,
                              __global int *reaction_index,
                              __global int *reaction_counts,
                              float dt)
{

    float4 polygon = hulls[polygon_id];
    float4 sensor  = hulls[sensor_id];
    int2 point_table = hull_point_tables[sensor_id];
    float4 point =  points[point_table.x];

    // get the sensor hull points and the polygon
    bool hit = point_polygon_containment(polygon_id, point.xy, hull_edge_tables, points, edges, edge_flags);

    if (!hit) return;

    hull_flags[sensor_id] |= SENSOR_HIT;

    float2 vert_hull_opposing = polygon.xy - sensor.xy;

    float displacement = calculateDisplacement(polygon_id, point.xy, hull_edge_tables, points, edges, edge_flags);

    int point_index = atomic_inc(&counter[0]);
    float8 vertex_reactions = (float8)((float2)(0.0f, displacement), vert_hull_opposing, (float2)(0.0f, 0.0f), (float2)(0.0f, 0.0f));
    reactions[point_index] = vertex_reactions;
    reaction_index[point_index] = point_table.x;
    atomic_inc(&reaction_counts[point_table.x]);

    printf("debug: line=%d poly=%d hit=%f", sensor_id, polygon_id, displacement);
    // 
}
