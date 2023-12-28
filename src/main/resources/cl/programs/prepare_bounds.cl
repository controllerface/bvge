/**
Prepares a vbo for rendering of bounding boxes as lines. The vbo will contain only a subset of 
the vertices that make up all bounding boxes, the start of the subset is defined by the offset value.
 */
__kernel void prepare_bounds(__global float4 *bounds, 
                             __global float2 *vbo,
                             int offset)
{
    int gid        = get_global_id(0);
    int bounds_id  = gid + offset;
    int vbo_offset = gid * 4;
    
    float4 bounding_box = bounds[bounds_id];
    
    vbo[vbo_offset++] = bounding_box.xy;
    vbo[vbo_offset++] = (float2)(bounding_box.x + bounding_box.z, bounding_box.y);
    vbo[vbo_offset++] = (float2)(bounding_box.x + bounding_box.z, bounding_box.y + bounding_box.w);
    vbo[vbo_offset++] = (float2)(bounding_box.x, bounding_box.y + bounding_box.w);
}
