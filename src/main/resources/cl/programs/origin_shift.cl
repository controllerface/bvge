__kernel void shift_points(__global float4 *points, 
                           float x_shift, 
                           float y_shift,
                           int max_point)
{
    int current_point = get_global_id(0);
    if (current_point >= max_point) return;
    float4 point = points[current_point];
    point.x -= x_shift;
    point.y -= y_shift;
    point.z -= x_shift;
    point.w -= y_shift;
    points[current_point] = point;
}

__kernel void shift_hulls(__global float4 *hulls, 
                          float x_shift, 
                          float y_shift,
                          int max_hull)
{
    int current_hull = get_global_id(0);
    if (current_hull >= max_hull) return;
    float4 hull = hulls[current_hull];
    hull.x -= x_shift;
    hull.y -= y_shift;
    hull.z -= x_shift;
    hull.w -= y_shift;
    hulls[current_hull] = hull;
}

__kernel void shift_entities(__global float4 *entities, 
                             float x_shift, 
                             float y_shift,
                             int max_entity)
{
    int current_entity = get_global_id(0);
    if (current_entity >= max_entity) return;
    float4 entity = entities[current_entity];
    entity.x -= x_shift;
    entity.y -= y_shift;
    entity.z -= x_shift;
    entity.w -= y_shift;
    entities[current_entity] = entity;
}