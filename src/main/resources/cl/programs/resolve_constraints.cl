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

    // calculate the normalized direction of separation
    float2 distance   = p2_v - p1_v;
    float  length     = fast_length(distance);
    float  difference = length - constraint;
    float2 direction  = fast_normalize(distance);

    direction.x *= difference * 0.5;
    direction.y *= difference * 0.5;

    p1_v = p1_v + direction;
    p2_v = p2_v - direction;

    p1.x = p1_v.x;
    p1.y = p1_v.y;
    p2.x = p2_v.x;
    p2.y = p2_v.y;

    points[p1_index] = p1;
    points[p2_index] = p2;
} 


inline void resolve_pin_constraint(__global float4 *hulls, 
                                   __global float4 *entities,
                                   __global float4 *points, 
                                   __global int2 *edges, 
                                   __global float *edge_lengths, 
                                   __global int *edge_pins, 
                                   int flags,
                                   int current_edge)
{
    int2 edge = edges[current_edge];
    float constraint = edge_lengths[current_edge];
    bool e_pin = (flags & E_SENSOR) != 0;

    float4 h = e_pin 
        ? entities[edge_pins[current_edge]]
        : hulls[edge_pins[current_edge]];

    float4 b = h;

    b.y = e_pin 
        ? h.y + constraint
        : h.y - constraint;

    b.w = e_pin 
        ? h.w + constraint
        : h.w - constraint;

    points[edge.x] = h;
    points[edge.y] = b;
} 

/**
Resolves edge constraints used for Verlet integration.
 */
__kernel void resolve_constraints(__global float4 *hulls,
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
            if (is_pin) resolve_pin_constraint(hulls, entities, points, edges, edge_lengths, edge_pins, flags, current_edge);
            else resolve_length_constraint(points, edges, edge_lengths, current_edge);
        }
    }
}
