__kernel void prepare_edges(__global float4 *points, 
                            __global float4 *edges,
                            __global float4 *vbo,
                            int offset)
{
    int gid = get_global_id(0); // this will be offset correctly by the kernel call
    int edge_id = gid + offset;
    float4 edge = edges[edge_id];
    int p1_index = (int)edge.s0;
    int p2_index = (int)edge.s1;
    float4 p1 = points[p1_index];
    float4 p2 = points[p2_index];
    float2 p1_v = p1.xy;
    float2 p2_v = p2.xy;
    float4 d = (float4)(p1_v, p2_v);
    vbo[gid] = d;
    if (gid >= 10000) printf("DEBUG: %d", gid);
}