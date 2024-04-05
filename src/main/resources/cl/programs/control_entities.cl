#define LEFT   0b0000000000000001
#define RIGHT  0b0000000000000010
#define UP     0b0000000000000100
#define DOWN   0b0000000000001000
#define JUMP   0b0000000000010000

__kernel void set_control_points(__global int *flags,
                                 __global int *indices,
                                 __global int *tick_budgets,
                                 __global float *linear_mag,
                                 __global float *jump_mag,
                                 int target,
                                 int new_flags, 
                                 int new_index, 
                                 int new_tick_budget, 
                                 float new_linear_mag, 
                                 float new_jump_mag)
{
    flags[target] = new_flags;
    indices[target] = new_index;
    linear_mag[target] = new_linear_mag;
    jump_mag[target] = new_jump_mag;

    int old_tick_budget = tick_budgets[target];
    tick_budgets[target] = new_tick_budget == -1 
        ? old_tick_budget 
        : new_tick_budget;
}

__kernel void handle_movement(__global float2 *armature_accel,
                              __global int *flags,
                              __global int *indices,
                              __global int *tick_budgets,
                              __global float *linear_mag,
                              __global float *jump_mag)
{
    int current_control_set = get_global_id(0);
    int current_flags = flags[current_control_set];
    int current_linear_mag = linear_mag[current_control_set];
    int current_jump_mag = jump_mag[current_control_set];
    int current_index = indices[current_control_set];
    float2 accel = armature_accel[current_index];

    bool is_mv_l = (current_flags & LEFT) !=0;
    bool is_mv_r = (current_flags & RIGHT) !=0;
    bool is_mv_u = (current_flags & UP) !=0;
    bool is_mv_d = (current_flags & DOWN) !=0;
    
    accel.x = is_mv_l 
        ? accel.x - current_linear_mag
        : accel.x;

    accel.x = is_mv_r 
        ? accel.x + current_linear_mag
        : accel.x;

    accel.y = is_mv_u 
        ? accel.y + current_linear_mag
        : accel.y;

    accel.y = is_mv_d 
        ? accel.y - current_linear_mag
        : accel.y;

    bool mv_jump = (current_flags & JUMP) !=0;

    accel.y = mv_jump 
        ? accel.y + current_jump_mag
        : accel.y;
}