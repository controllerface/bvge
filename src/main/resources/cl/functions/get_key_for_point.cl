// calculates a spatial index cell for a given point
inline int2 get_key_for_point(float px, float py,
                    float x_spacing, float y_spacing,
                    float x_origin, float y_origin)
{
    int2 out;
    float adjusted_x = px - (x_origin);
    float adjusted_y = py - (y_origin);
    int index_x = ((int) floor( native_divide(adjusted_x, x_spacing) ));
    int index_y = ((int) floor( native_divide(adjusted_y, y_spacing) ));
    out.x = index_x;
    out.y = index_y;
    return out;
}
