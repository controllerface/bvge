#define IDLE         0
#define WALKING      1
#define RUNNING      2
#define FALLING_FAST 3
#define JUMP_START   4
#define JUMPING      5
#define IN_AIR       6
#define LAND_HARD    7
#define FALLING_SLOW 8
#define LAND_SOFT    9
#define SWIM_UP      10

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
                              __global short2 *armature_motion_states,
                              __global int *armature_flags,
                              __global int2 *armature_animation_indices,
                              __global float2 *armature_animation_elapsed,
                              __global float2 *armature_animation_blend,
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
    int2 anim_index = armature_animation_indices[current_index];

    bool is_mv_l = (current_flags & LEFT) !=0;
    bool is_mv_r = (current_flags & RIGHT) !=0;
    bool is_mv_u = (current_flags & UP) !=0;
    bool is_mv_d = (current_flags & DOWN) !=0;
    bool mv_jump = (current_flags & JUMP) !=0;

    float threshold = 10.0f;
    float2 vel = (armature.xy - armature.zw) / dt;

    // can jump?
    bool can_jump = (arm_flag & CAN_JUMP) !=0;
    bool is_wet = (arm_flag & IS_WET) !=0;

    int tick_slice = 0;

    short2 motion_state = armature_motion_states[current_index];
    
    float2 current_time = armature_animation_elapsed[current_index];

    int b_reset = is_wet ? 10 : 20;

    current_budget = can_jump && !mv_jump
        ? b_reset
        : current_budget;

    int next_state = anim_index.x;
    switch(anim_index.x)
    {
        case IDLE:
            if (is_mv_l || is_mv_r) next_state = WALKING;
            if (can_jump && current_budget > 0 && mv_jump) next_state = JUMP_START;
            if (motion_state.x > 50) next_state = FALLING_SLOW;
            if (motion_state.y > 50) next_state = is_wet ? SWIM_UP : IN_AIR;
            if (next_state != IDLE)
            {
                float t = next_state == JUMP_START 
                    ? 0.1f
                    : 0.4f;
                current_time.y = current_time.x;
                anim_index.y = anim_index.x;
                armature_animation_blend[current_index] = (float2)(t, 0.0f);
            }
            break;

        case WALKING: 
            if (!is_mv_l && !is_mv_r) next_state = IDLE;
            if (can_jump && current_budget > 0 && mv_jump) next_state = JUMP_START;
            if (motion_state.x > 50) next_state = FALLING_SLOW;
            if (motion_state.y > 50) next_state = is_wet ? SWIM_UP : IN_AIR;
            if (next_state != WALKING)
            {
                float t = next_state == JUMP_START 
                    ? 0.1f
                    : 0.2f;
                current_time.y = current_time.x;
                anim_index.y = anim_index.x;
                armature_animation_blend[current_index] = (float2)(t, 0.0f);
            }
            break;

        case RUNNING:
            break;

        case FALLING_SLOW:
            if (can_jump) next_state = motion_state.x > 200 
                ? LAND_HARD 
                : LAND_SOFT;
            if (motion_state.x > 200) next_state = FALLING_FAST;
            if (motion_state.y > 50) next_state = is_wet ? SWIM_UP : IN_AIR;
            if (next_state != FALLING_SLOW)
            {
                float t = next_state == LAND_HARD || next_state == LAND_SOFT 
                    ? 0.1f
                    : 0.2f;
                current_time.y = current_time.x;
                anim_index.y = anim_index.x;
                armature_animation_blend[current_index] = (float2)(t, 0.0f);
            }
            break;

        case FALLING_FAST:
            if (can_jump) next_state = motion_state.x > 200 
                ? LAND_HARD 
                : LAND_SOFT;
            if (motion_state.x < 200) next_state = FALLING_SLOW;
            if (motion_state.y > 50) next_state = is_wet ? SWIM_UP : IN_AIR;
            if (next_state != FALLING_FAST)
            {
                float t = next_state == LAND_HARD || next_state == LAND_SOFT 
                    ? 0.1f
                    : 0.2f;
                current_time.y = current_time.x;
                anim_index.y = anim_index.x;
                armature_animation_blend[current_index] = (float2)(t, 0.0f);
            }
            break;

        case JUMP_START:
            if (current_time.x > 0.15f) next_state = JUMPING;
            break;

        case JUMPING:
            tick_slice = current_budget > 0 ? 1 : 0;
            current_budget -= tick_slice;
            float jump_amount = tick_slice == 1 
                ? mv_jump 
                    ? current_jump_mag 
                    : current_jump_mag / 2
                : 0;
            accel.y = jump_amount;
            if (tick_slice == 0) next_state = is_wet ? SWIM_UP : IN_AIR;
            if (next_state != JUMPING)
            {
                float t = 0.5f;
                current_time.y = current_time.x;
                anim_index.y = anim_index.x;
                armature_animation_blend[current_index] = (float2)(t, 0.0f);
            }
            break;

        case IN_AIR:
            if (can_jump) next_state = motion_state.x > 200 
                ? LAND_HARD 
                : LAND_SOFT;
            if (motion_state.x > 50) next_state = FALLING_SLOW;
            if (next_state != IN_AIR)
            {
                float t = next_state == LAND_HARD || next_state == LAND_SOFT 
                    ? 0.1f
                    : 0.2f;
                current_time.y = current_time.x;
                anim_index.y = anim_index.x;
                armature_animation_blend[current_index] = (float2)(t, 0.0f);
            }
            break;
        
        case SWIM_UP:
            if (can_jump) next_state = motion_state.x > 200 
                ? LAND_HARD 
                : LAND_SOFT;
            if (motion_state.x > 50) next_state = FALLING_SLOW;
            if (next_state != SWIM_UP)
            {
                float t = next_state == LAND_HARD || next_state == LAND_SOFT 
                    ? 0.1f
                    : 0.2f;
                current_time.y = current_time.x;
                anim_index.y = anim_index.x;
                armature_animation_blend[current_index] = (float2)(t, 0.0f);
            }
            break;

        case LAND_SOFT:
            if (current_time.x > 0.08f) next_state = IDLE;
            break;

        case LAND_HARD:
            if (current_time.x > 0.22f) next_state = IDLE;
            break;
    }

    if (anim_index.x != next_state) 
    {
        // current_time.y = current_time.x;
        // anim_index.y = anim_index.x;
        // armature_animation_blend[current_index] = (float2)(0.3, 0.0f);
        current_time.x = 0.0f;
        armature_animation_elapsed[current_index] = current_time;
    }
    anim_index.x = next_state;

    motion_state.x = (vel.y < -threshold) 
        ? motion_state.x + 1 
        : 0;

    motion_state.y = (vel.y > threshold) 
        ? motion_state.y + 1 
        : 0;

    motion_state.x = motion_state.x > 1000 ? 1000 : motion_state.x;
    motion_state.y = motion_state.y > 1000 ? 1000 : motion_state.y;

    armature_motion_states[current_index] = motion_state;


    arm_flag = is_mv_l != is_mv_r 
        ? is_mv_l
            ? arm_flag | FACE_LEFT 
            : is_mv_r 
                ? arm_flag & ~FACE_LEFT 
                : arm_flag
        : arm_flag; 

    
    float w_mod = is_wet
        ? 0.5f
        : 1.0f;

    // update left/right movement
    accel.x = is_mv_l && !is_mv_r
        ? -current_linear_mag * w_mod
        : accel.x;

    accel.x = is_mv_r && !is_mv_l
        ? current_linear_mag * w_mod
        : accel.x;

    // upward and downward movement is disabled unless in water

    accel.y = is_mv_u && is_wet
        ? current_linear_mag * 1.5
        : accel.y;

    accel.y = is_mv_d && is_wet
        ? -current_linear_mag
        : accel.y;

    tick_budgets[current_control_set] = current_budget;
    armature_accel[current_index] = accel;
    armature_flags[current_index] = arm_flag;
    armature_animation_indices[current_index] = anim_index;
}
