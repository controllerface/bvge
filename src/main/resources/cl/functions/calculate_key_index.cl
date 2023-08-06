/**
Calculates a key index within a uniform grid spatial partition. 
The grid is assumed to be laid out in row major order.
 */
inline int calculate_key_index(int x_subdivisions, int x, int y)
{
    return x_subdivisions * y + x;
}
