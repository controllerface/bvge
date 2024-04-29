inline float2 calculate_centroid(__global const float4 *points, int2 point_table)
{
    float2 result;

    int start = point_table.x;
    int end   = point_table.y;
	int vert_count = end - start + 1;

    float x_sum = 0;
    float y_sum = 0;

    for (int i = 0; i < vert_count; i++)
    {
        int n = start + i;
        float2 v = points[n].xy;
        x_sum += v.x;
        y_sum += v.y;
    }

    float x = native_divide(x_sum, vert_count);
    float y = native_divide(y_sum, vert_count);
    result.x = x;
    result.y = y;

    return result;
}
