#define LEFT   0b0000000000000001
#define RIGHT  0b0000000000000010
#define UP     0b0000000000000100
#define DOWN   0b0000000000001000
#define JUMP   0b0000000000010000

__kernel void set_control_points(__global int *control_flags,
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
    control_flags[target] = new_flags;
    indices[target] = new_index;
    linear_mag[target] = new_linear_mag;
    jump_mag[target] = new_jump_mag;

    int old_tick_budget = tick_budgets[target];
    tick_budgets[target] = new_tick_budget == -1 
        ? old_tick_budget 
        : new_tick_budget;
}

__kernel void handle_movement(__global float2 *armature_accel,
                              __global int4 *armature_flags,
                              __global int *control_flags,
                              __global int *indices,
                              __global int *tick_budgets,
                              __global float *linear_mag,
                              __global float *jump_mag)
{
    int current_control_set = get_global_id(0);
    int current_flags = control_flags[current_control_set];
    int current_budget = tick_budgets[current_control_set];
    float current_linear_mag = linear_mag[current_control_set];
    float current_jump_mag = jump_mag[current_control_set];

    int current_index = indices[current_control_set];
    float2 accel = armature_accel[current_index];
    int4 arm_flag = armature_flags[current_index];

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

    // todo: upward and downward movement is disabled so jumping can work correctly,
    //  but may be worth doing some checks later to re-enbale this depending on circulmstances
    //  for example swimming, or zero-G, etc.

    // accel.y = is_mv_u 
    //     ? accel.y + current_linear_mag
    //     : accel.y;

    // accel.y = is_mv_d 
    //     ? accel.y - current_linear_mag
    //     : accel.y;

    bool mv_jump = (current_flags & JUMP) !=0;

    int tick_slice = current_budget > 0 
        ? 1 
        : 0;

    // todo: logic should work so that if jump is released but ground hasn't been touched,
    //  the budget is set to 0. The idea is to make it so a short hop still results in
    //  the whole budget being taken even though it wasn't all used. then it is reset
    //  when a ground touch occurs. Otherwise, two small jumps from mid-air would be possible
    //  which should be avoided.

    current_budget = mv_jump 
        ? current_budget - tick_slice 
        : current_budget;

    float jump_amount = mv_jump && tick_slice == 1
        ? current_jump_mag
        : 0;

    accel.y = mv_jump 
        ? jump_amount
        : accel.y;

    tick_budgets[current_control_set] = current_budget;
    armature_accel[current_index] = accel;
}