/**
Resolves edge constraints used for Verlet integration.
 */
__kernel void resolve_constraints(__global int2 *hull_edge_tables,
                                  __global int2 *bounds_bank_data,
                                  __global float4 *points,
                                  __global int2 *edges,
                                  __global float *edge_lengths,
                                  int process_all,
                                  int max_hull)
{
    int current_hull = get_global_id(0);
    if (current_hull >= max_hull) return;

    // the element table contains the relevant pointers into the edge buffer, and
    // the bounding box is used to check if the edges should be processed.
    int2 edge_table = hull_edge_tables[current_hull];
    int2 bounds_bank = bounds_bank_data[current_hull];

    // extract the bank size from the boundary. Hulls with empty banks are out of bounds
    int bank_size = bounds_bank.y;

    // we usually only want to process objects that are in bounds, however in order to ensure 
    // simulation stability, out of bounds objects need at least one update per frame.
    if (bank_size > 0 || process_all == 1)
    {
        // get the starting and ending edges for this hull
        int start_edge = edge_table.x;
        int end_edge = edge_table.y;

        // for each edge, we need to calculate the current distance between points,
        // and then move the vertices apart so they meet the length requirement.
        //for (int current_edge = end_edge; current_edge >= start_edge; current_edge--)
        for (int current_edge = start_edge; current_edge <= end_edge; current_edge++)
        {
            // get this edge
            int2 edge = edges[current_edge];
            
            // grab the point indices from the edge object
            int p1_index = edge.x;
            int p2_index = edge.y;
            float constraint = edge_lengths[current_edge];
            
            // get the points for this edge
            float4 p1 = points[p1_index];
            float4 p2 = points[p2_index];
            
            // extract just the current vertex info for processing
            float2 p1_v = p1.xy;
            float2 p2_v = p2.xy;




    // float2 p1_tail = p1.zw;
    // float p1_dist = fast_distance(p1.xy, p1.zw);

    // float2 p2_tail = p2.zw;
    // float p2_dist = fast_distance(p2.xy, p2.zw);




            // calculate the normalized direction of separation
            float2 sub = p2_v - p1_v;
            float len = fast_length(sub);
            float diff = len - constraint;
            float2 direction = fast_normalize(sub);
        
            // the difference is halved and the direction is set to that magnitude
            direction.x *= diff * 0.5;
            direction.y *= diff * 0.5;
        
            // move the first vertex in the positive direction, move the second negative
            p1_v = p1_v + direction;
            p2_v = p2_v - direction;

            // store the updated values in the points
            p1.x = p1_v.x;
            p1.y = p1_v.y;
            p2.x = p2_v.x;
            p2.y = p2_v.y;



    // using the initial data, compared to the new position, calculate the updated previous
    // position to ensure it is equivalent to the initial position delta. This preserves 
    // velocity.
    // float2 p1_offset = p1.xy - p1_tail;
    // float2 p2_offset = p2.xy - p2_tail;
    // float new_len_1 = fast_length(p1_offset);
    // float new_len_2 = fast_length(p2_offset);

    // p1_offset = new_len_1 == 0.0f 
    //     ? p1_offset 
    //     : native_divide(p1_offset, new_len_1);

    // p2_offset = new_len_2 == 0.0f 
    //     ? p2_offset 
    //     : native_divide(p2_offset, new_len_2);

    // p1.zw = p1.xy - p1_dist * p1_offset;
    // p2.zw = p2.xy - p2_dist * p2_offset;



            // set the updated points into the buffer
            points[p1_index] = p1;
            points[p2_index] = p2;
        }
    }
}
