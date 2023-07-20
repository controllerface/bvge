inline float3 project_polygon(__global const float4 *points, float16 body, float2 normal)
{
    int start = (int)body.s7;
    int end   = (int)body.s8;
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