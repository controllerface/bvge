
inline int compare(float a, float b) 
{
	return (a > b ? 1 : 0) - (a < b ? 1 : 0);
}

inline int orientation(float x0, float y0, float x1, float y1, float x2, float y2) 
{
	return compare((x2 - x1) * (y1 - y0), (x1 - x0) * (y2 - y1));
}

inline bool linesIntersect(float4 line1, float4 line2) 
{
	return orientation(line1.x, line1.y, line1.z, line1.w, line2.x, line2.y) != orientation(line1.x, line1.y, line1.z, line1.w, line2.z, line2.w) 
        && orientation(line2.x, line2.y, line2.z, line2.w, line1.x, line1.y) != orientation(line2.x, line2.y, line2.z, line2.w, line1.z, line1.w);
}

float2 projectPointOnLine(float2 p1, float2 p2, float2 C) 
{
    float dx = p2.x - p1.x;
    float dy = p2.y - p1.y;
    float lengthSquared = dx * dx + dy * dy;

    if (lengthSquared == 0) {
        return (float2)(0.0f);
    }

    float t = ((C.x - p1.x) * dx + (C.y - p1.y) * dy) / lengthSquared;
    t = max(0.0f, min(1.0f, t)); // Clamp t to [0, 1]

    return (float2)(p1.x + t * dx, p1.y + t * dy);
}

 float calculateLerpValue(float2 p1, float2 p2, float2 C) 
 {
    float2 projection = projectPointOnLine(p1, p2, C);
    float totalDistance = distance(p1, p2);
    float distanceToProjection = distance(p1, projection);

    // Ensure no division by zero
    if (totalDistance == 0) 
    {
        return 0;
    }

    return distanceToProjection / totalDistance;



    // float totalDistance = distance(p1, p2);
    // float distanceToC = distance(p1, C);
        
    // // Ensure no division by zero
    // if (totalDistance == 0) 
    // {
    //     return 0;
    // }
        
    // return distanceToC / totalDistance;
}

float2 getIntersection(float4 line1, float4 line2) 
{
    // Line AB represented as a1x + b1y = c1
    float a1 = line1.w - line1.y;
    float b1 = line1.x - line1.z;
    float c1 = a1*(line1.x) + b1*(line1.y);
    
    // Line CD represented as a2x + b2y = c2
    float a2 = line2.w - line2.y;
    float b2 = line2.x - line2.z;
    float c2 = a2*(line2.x)+ b2*(line2.y);
    
    float determinant = a1*b2 - a2*b1;
    
    if (determinant == 0)
    {
        // The lines are parallel. This is simplified
        // by returning a pair of FLT_MAX
        return (float2)(0.0f);
    }
    else
    {
        float x = (b2*c1 - b1*c2)/determinant;
        float y = (a1*c2 - a2*c1)/determinant;
        return (float2)(x, y);
    }
}


/**
Performs axis-aligned bounding box collision detection as part of a broad phase collision step.
 */
__kernel void ccd_collide(__global int2 *edges,
                          __global float4 *points,
                          __global float *point_anti_time,
                          __global int *edge_flags,
                          __global int2 *candidates,
                          int max_index)
{
    int gid = get_global_id(0);
    if (gid >= max_index) return;

    int2 current_pair = candidates[gid];
    
    int edge1_id = current_pair.x;
    int edge2_id = current_pair.y;
    
    int edge1_flags = edge_flags[edge1_id];
    int edge2_flags = edge_flags[edge2_id];
    
    bool edge1_static = (edge1_flags & IS_STATIC) !=0;
    bool edge2_static = (edge2_flags & IS_STATIC) !=0;

    //printf("edge 1: %d edge 2: %d", index, next);
    int static_edge_id = edge1_static 
        ? edge1_id 
        : edge2_id;

    int non_static_edge_id = edge2_static 
        ? edge1_id 
        : edge2_id;

    int2 static_edge = edges[static_edge_id];
    int2 non_static_edge = edges[non_static_edge_id];
    
    float4 static_line = (float4)(points[static_edge.x].xy, points[static_edge.y].xy);
    float4 point1 = points[non_static_edge.x];
    float4 point2 = points[non_static_edge.y];

    if (linesIntersect(static_line, point1))
    {
        float2 intersection = getIntersection(static_line, point1);
        float dist = calculateLerpValue(point1.zw, point1.xy, intersection);
        point_anti_time[non_static_edge.x] = dist;
        // printf("lerp: %f static edge: %f,%f -> %f,%f dyn edge 1: %f,%f -> %f,%f intersection: %f,%f", 
        //     dist, 
        //     static_line.x, static_line.y, static_line.z, static_line.w, 
        //     point1.x, point1.y, point1.z, point1.w, 
        //     intersection.x, intersection.y);
    }
    if (linesIntersect(static_line, point2))
    {
        float2 intersection = getIntersection(static_line, point2);
        float dist = calculateLerpValue(point2.zw, point2.xy, intersection);
        point_anti_time[non_static_edge.y] = dist;
        // printf("lerp: %f static edge: %f,%f -> %f,%f dyn edge 2: %f,%f -> %f,%f intersection: %f,%f",
        //     dist, 
        //     static_line.x, static_line.y, static_line.z, static_line.w, 
        //     point2.x, point2.y, point2.z, point2.w, 
        //     intersection.x, intersection.y);
    }
}

__kernel void ccd_react(__global float4 *points,
                        __global float *point_anti_time,
                        int max_point)
{
    int current_point = get_global_id(0);
    if (current_point >= max_point) return;
    float rewind = point_anti_time[current_point];
    if (rewind == 0.0f) return;
    point_anti_time[current_point] = 0.0f;
    float4 point = points[current_point];
    float2 rewound = float2_lerp(point.xy, point.zw, rewind);
    float2 diff = point.xy - rewound;
    point.zw = point.xy;
    point.xy = rewound;
    points[current_point] = point;
}
