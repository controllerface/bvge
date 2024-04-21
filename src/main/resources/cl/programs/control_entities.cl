#define IDLE       0
#define WALKING    1
#define RUNNING    2
#define FALLING    3
#define JUMP_START 4
#define JUMPING    5
#define IN_AIR     6
#define LANDING    7

__kernel void set_control_points(__global int *control_flags,
                                 __global int *indices,
                                 __global float *linear_mag,
                                 __global float *jump_mag,
                                 int target,
                                 int new_flags, 
                                 int new_index, 
                                 float new_linear_mag, 
                                 float new_jump_mag)
{
    control_flags[target] = new_flags;
    indices[target] = new_index;
    linear_mag[target] = new_linear_mag;
    jump_mag[target] = new_jump_mag;
}

__kernel void handle_movement(__global float4 *armatures,
                              __global float2 *armature_accel,
                              __global short2 *armature_animation_states,
                              __global int *armature_flags,
                              __global int *armature_animation_indices,
                              __global float *armature_animation_elapsed,
                              __global int *control_flags,
                              __global int *indices,
                              __global int *tick_budgets,
                              __global float *linear_mag,
                              __global float *jump_mag,
                              float dt)
{
    int current_control_set = get_global_id(0);
    int current_flags = control_flags[current_control_set];
    int current_budget = tick_budgets[current_control_set];
    float current_linear_mag = linear_mag[current_control_set];
    float current_jump_mag = jump_mag[current_control_set];

    int current_index = indices[current_control_set];
    float4 armature = armatures[current_index];
    float2 accel = armature_accel[current_index];
    int arm_flag = armature_flags[current_index];
    int anim_state = armature_animation_indices[current_index];

    bool is_mv_l = (current_flags & LEFT) !=0;
    bool is_mv_r = (current_flags & RIGHT) !=0;
    bool is_mv_u = (current_flags & UP) !=0;
    bool is_mv_d = (current_flags & DOWN) !=0;
    bool mv_jump = (current_flags & JUMP) !=0;


    float threshold = 10.0f;
    float2 vel = (armature.xy - armature.zw) / dt;

    // update left/right movement
    accel.x = is_mv_l && !is_mv_r
        ? -current_linear_mag
        : accel.x;

    accel.x = is_mv_r && !is_mv_l
        ? current_linear_mag
        : accel.x;

    // can jump?
    bool can_jump = (arm_flag & CAN_JUMP) !=0;

    int tick_slice = 0;

    short2 anim_s = armature_animation_states[current_index];
    
    float ct = armature_animation_elapsed[current_index];
    int next_state = anim_state;
    switch(anim_state)
    {
        case IDLE:
            current_budget = can_jump && !mv_jump
                ? 40
                : current_budget;
            if (is_mv_l || is_mv_r) next_state = WALKING;
            if (can_jump && current_budget > 0 && mv_jump) next_state = JUMP_START;
            if (anim_s.x > 100) next_state = FALLING;
            if (anim_s.y > 50) next_state = IN_AIR;
            break;

        case WALKING: 
            current_budget = can_jump && !mv_jump
                ? 40
                : current_budget;
            if (!is_mv_l && !is_mv_r) next_state = IDLE;
            if (can_jump && current_budget > 0 && mv_jump) next_state = JUMP_START;
            if (anim_s.x > 100) next_state = FALLING;
            if (anim_s.y > 50) next_state = IN_AIR;
            break;

        case RUNNING:
            break;

        case FALLING:
            if (can_jump) next_state = anim_s.x > 200 ? LANDING : IDLE;
            if (anim_s.y > 50) next_state = IN_AIR;
            break;

        case JUMP_START:
            if (ct > 0.08f) next_state = JUMPING;
            break;

        case JUMPING:
            tick_slice = current_budget > 0 
                ? 1 
                : 0;

            current_budget -= tick_slice;

            float jump_amount = mv_jump && tick_slice == 1
                ? current_jump_mag
                : 0;

            accel.y = current_jump_mag;
            if (tick_slice == 0) next_state = IN_AIR;
            break;

        case IN_AIR:
            if (can_jump) next_state = anim_s.x > 200 ? LANDING : IDLE;
            if (anim_s.x > 50) next_state = FALLING;
            break;

        case LANDING:
            if (ct > 0.26f) next_state = IDLE;
            break;

    }

    if (anim_state != next_state) armature_animation_elapsed[current_index] = 0.0f;
    anim_state = next_state;

    anim_s.x = (vel.y < -threshold) 
        ? anim_s.x + 1 
        : 0;

    anim_s.y = (vel.y > threshold) 
        ? anim_s.y + 1 
        : 0;

    anim_s.x = anim_s.x > 1000 ? 1000 : anim_s.x;
    anim_s.y = anim_s.y > 1000 ? 1000 : anim_s.y;

    armature_animation_states[current_index] = anim_s;


    // todo: upward and downward movement is disabled so jumping can work correctly,
    //  but may be worth doing some checks later to re-enbale this depending on circulmstances
    //  for example swimming, or zero-G, etc.

    // accel.y = is_mv_u 
    //     ? current_linear_mag
    //     : accel.y;

    // accel.y = is_mv_d 
    //     ? -current_linear_mag
    //     : accel.y;


    tick_budgets[current_control_set] = current_budget;
    armature_accel[current_index] = accel;
    armature_flags[current_index] = arm_flag;
    armature_animation_indices[current_index] = anim_state;

}