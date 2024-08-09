inline void resolve_length_constraint(__global float4 *points, __global int2 *edges, __global float *edge_lengths, int current_edge)
{
    int2 edge = edges[current_edge];

    int p1_index = edge.x;
    int p2_index = edge.y;
    float constraint = edge_lengths[current_edge];

    float4 p1 = points[p1_index];
    float4 p2 = points[p2_index];

    float2 p1_v = p1.xy;
    float2 p2_v = p2.xy;

    float2 p1_p = p1.zw;
    float2 p2_p = p2.zw;

    // calculate the normalized direction of separation
    float2 distance     = p2_v - p1_v;
    float2 distance_p   = p2_p - p1_p;
    float  length       = fast_length(distance);
    float  length_p     = fast_length(distance_p);
    float  difference   = length - constraint;
    float  difference_p = length_p - constraint;
    float2 direction    = fast_normalize(distance);
    float2 direction_p  = fast_normalize(distance_p);

    direction.x *= difference * 0.5;
    direction.y *= difference * 0.5;

    direction_p.x *= difference_p * 0.5;
    direction_p.y *= difference_p * 0.5;

    p1_v = p1_v + direction;
    p2_v = p2_v - direction;

    p1_p = p1_p + direction_p;
    p2_p = p2_p - direction_p;

    p1.x = p1_v.x;
    p1.y = p1_v.y;
    p2.x = p2_v.x;
    p2.y = p2_v.y;

    p1.z = p1_p.x;
    p1.w = p1_p.y;
    p2.z = p2_p.x;
    p2.w = p2_p.y;

    points[p1_index] = p1;
    points[p2_index] = p2;
} 

inline void resolve_pin_constraint(__global float4 *hulls, 
                                   __global float4 *points, 
                                   __global int2 *edges, 
                                   __global float *edge_lengths, 
                                   __global int *edge_pins, 
                                   int current_edge)
{
    int2 edge = edges[current_edge];
    float constraint = edge_lengths[current_edge];
    float4 h = hulls[edge_pins[current_edge]];
    float4 b = h;
    b.y = h.y - constraint;
    b.w = h.w - constraint;
    points[edge.x] = h;
    points[edge.y] = b;
} 

inline void resolve_e_pin_constraint(__global float4 *entities,
                                     __global float4 *points, 
                                     __global int2 *edges, 
                                     __global float *edge_lengths, 
                                     __global int *edge_pins, 
                                     int current_edge)
{
    int2 edge = edges[current_edge];
    float constraint = edge_lengths[current_edge];
    float4 h = entities[edge_pins[current_edge]];
    float4 b = h;
    b.y = h.y + constraint;
    b.w = h.w + constraint;
    //h.y -= 16;
    points[edge.x] = h;
    points[edge.y] = b;
} 

/**
Resolves edge constraints used for Verlet integration.
 */
__kernel void resolve_constraints(__global float4 *hulls,
                                  __global int *hull_flags,
                                  __global float4 *entities,
                                  __global int2 *hull_edge_tables,
                                  __global int2 *bounds_bank_data,
                                  __global float4 *points,
                                  __global int2 *edges,
                                  __global float *edge_lengths,
                                  __global int *edge_flags,
                                  __global int *edge_pins,
                                  int process_all,
                                  int max_hull)
{
    int current_hull = get_global_id(0);
    if (current_hull >= max_hull) return;
    int flags = hull_flags[current_hull];
    if ((flags & IS_STATIC) != 0) return;
    int2 edge_table = hull_edge_tables[current_hull];
    int2 bounds_bank = bounds_bank_data[current_hull];
    int bank_size = bounds_bank.y;
    if (bank_size > 0 || process_all == 1)
    {
        for (int current_edge = edge_table.x; current_edge <= edge_table.y; current_edge++)
        {
            // todo: handle sensor edges with different logic
            int flags = edge_flags[current_edge];
            bool is_pin = (flags & SENSOR_EDGE) != 0;
            bool e_pin = (flags & E_SENSOR) != 0;
            if (e_pin) resolve_e_pin_constraint(entities, points, edges, edge_lengths, edge_pins, current_edge);
            else if (is_pin) resolve_pin_constraint(hulls, points, edges, edge_lengths, edge_pins, current_edge);
            else resolve_length_constraint(points, edges, edge_lengths, current_edge);
        }
    }
}
