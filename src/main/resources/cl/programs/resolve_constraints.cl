/**
Resolves edge constraints used for Verlet integration.
 */
__kernel void resolve_constraints(__global int4 *element_tables,
                                  __global int2 *bounds_bank_data,
                                  __global float4 *points,
                                  __global int2 *edges,
                                  __global float *edge_lengths,
                                  int process_all)
{
    int gid = get_global_id(0);
    
    // the element table contains the relevant pointers into the edge buffer, and
    // the bounding box is used to check if the edges should be processed.
    int4 element_table = element_tables[gid];
    int2 bounds_bank = bounds_bank_data[gid];

    // extract the bank size from the boundary. Hulls with empty banks are out of bounds
    int bank_size = bounds_bank.y;

    // we usually only want to process objects that are in bounds, however in order to ensure 
    // simulation stability, out of bounds objects need at least one update per frame.
    if (bank_size > 0 || process_all == 1)
    {
        // get the starting and ending edges for this hull
        int start_edge = element_table.z;
        int end_edge = element_table.w;

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

            // set the updated points into the buffer
            points[p1_index] = p1;
            points[p2_index] = p2;
        }
    }
}