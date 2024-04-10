
/**
Projects a circle onto the given normal vector. The return vector contains 
the min/max distances only, there is no index for circle projection, but 
this function returns a float3 so results can be used together with the 
output of project_polygon.
 */
inline float3 project_circle(float2 circle, float circle_size, float2 normal)
{
    float3 result;
    result.x = (float)0; // min
    result.y = (float)0; // max
    result.z = (float)0; // index (unused)

    float2 unit = fast_normalize(normal);
    float2 dirRad = unit * native_divide(circle_size, 2);
    float2 p1 = circle.xy + dirRad;
    float2 p2 = circle.xy - dirRad;

    float min = dot(p1, normal);
    float max = dot(p2, normal);
    bool invert = false;
    if(min > max)
    {
        float t = min;
        min = max;
        max = t;
        invert = true;
    }
    result.x = min;
    result.y = max;
    return result;
}