/**
Calculates the extents of an object within the space partition, using the 4 spatial keys
generated for the object. Note that the extents may fall inside, outside, or both, but 
generally speaking if extents overlap the current spatial partition 
boundary, they are considered "in bounds".
 */
inline int4 getExtents(int2 corners[])
{
    int4 r;
    r.x = INT_MAX; // min_x
    r.y = INT_MIN; // max_x
    r.z = INT_MAX; // min_y
    r.w = INT_MIN; // max_y
    for (int i = 0; i < sizeof(corners); i++)
    {
        int2 corner = corners[i];
        if (corner.x < r.x)
        {
            r.x = corner.x;
        }
        if (corner.x > r.y)
        {
            r.y = corner.x;
        }
        if (corner.y < r.z)
        {
            r.z = corner.y;
        }
        if (corner.y > r.w)
        {
            r.w = corner.y;
        }
    }
    return r;
}
