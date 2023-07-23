__kernel void update_accel(__global float16 *bodies,
                           int target,
                           float2 new_value)
{
    int gid = get_global_id(0);
    float16 body = bodies[target];
    body.s4 = new_value.x;
   	body.s5 = new_value.y;
    bodies[target] = body;
}