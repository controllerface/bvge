typedef struct 
{
    bool is_mv_l;
    bool is_mv_r;
    bool mv_jump;
    bool mv_run;
    bool can_jump;
    bool is_wet;
    bool is_click_1;
    bool is_click_2;
    int current_budget;
    short2 motion_state;
    float current_time;
    int anim_index;
    float jump_mag;
} InputState;

typedef struct
{
    bool accel;
    bool attack;
    int next_state;
    int next_budget;
    float jump_amount;
} OutputState;

OutputState init_output(int current_state)
{
    OutputState output = { false, false, current_state, 0, 0.0f };
    return output;
}

OutputState idle_state(InputState input)
{
    OutputState output = init_output(IDLE);
    if (input.is_mv_l || input.is_mv_r) output.next_state = input.mv_run ? RUNNING : WALKING;
    if (input.is_click_1) output.next_state = PUNCH;
    if (input.can_jump && input.current_budget > 0 && input.mv_jump) output.next_state = RECOIL;
    if (input.motion_state.x > 100) output.next_state = input.is_wet ? SWIM_DOWN : FALLING_SLOW;
    if (input.motion_state.y > 150) output.next_state = input.is_wet ? SWIM_UP : IN_AIR;
    return output;
}

OutputState walking_state(InputState input)
{
    OutputState output = init_output(WALKING);
    if (input.mv_run) output.next_state = RUNNING;
    if (!input.is_mv_l && !input.is_mv_r) output.next_state = IDLE;
    if (input.is_click_1) output.next_state = PUNCH;
    if (input.can_jump && input.current_budget > 0 && input.mv_jump) output.next_state = RECOIL;
    if (input.motion_state.x > 100) output.next_state = input.is_wet ? SWIM_DOWN : FALLING_SLOW;
    if (input.motion_state.y > 150) output.next_state = input.is_wet ? SWIM_UP : IN_AIR;
    return output;
}

OutputState running_state(InputState input)
{
    OutputState output = init_output(RUNNING);
    if (!input.mv_run) output.next_state = WALKING;
    if (!input.is_mv_l && !input.is_mv_r) output.next_state = IDLE;
    if (input.is_click_1) output.next_state = PUNCH;
    if (input.can_jump && input.current_budget > 0 && input.mv_jump) output.next_state = RECOIL;
    if (input.motion_state.x > 100) output.next_state = input.is_wet ? SWIM_DOWN : FALLING_SLOW;
    if (input.motion_state.y > 150) output.next_state = input.is_wet ? SWIM_UP : IN_AIR;
    return output;
}

OutputState falling_slow_state(InputState input)
{
    OutputState output = init_output(FALLING_SLOW);
    if (input.can_jump) output.next_state = input.motion_state.x > 200 
        ? LAND_HARD 
        : LAND_SOFT;
    if (input.motion_state.x > 200) output.next_state = input.is_wet ? SWIM_DOWN : FALLING_FAST;
    if (input.motion_state.y > 50) output.next_state = input.is_wet ? SWIM_UP : IN_AIR;
    return output;
}

OutputState falling_fast_state(InputState input)
{
    OutputState output = init_output(FALLING_FAST);
    if (input.can_jump) output.next_state = input.motion_state.x > 200 
        ? LAND_HARD 
        : LAND_SOFT;
    if (input.is_wet) output.next_state = SWIM_DOWN;
    if (input.motion_state.x < 200) output.next_state = input.is_wet ? SWIM_DOWN : FALLING_SLOW;
    if (input.motion_state.y > 50) output.next_state = input.is_wet ? SWIM_UP : IN_AIR;
    return output;
}

OutputState recoil_state(InputState input)
{
    OutputState output = init_output(RECOIL);
    if (input.current_time > 0.15f) output.next_state = JUMPING;
    return output;
}

OutputState jumping_state(InputState input)
{
    OutputState output = init_output(JUMPING);
    output.accel = true;
    output.next_budget = input.current_budget;
    int tick_slice = input.current_budget > 0 ? 1 : 0;
    output.next_budget -= tick_slice;
    output.jump_amount = tick_slice == 1 
        ? input.mv_jump 
            ? input.jump_mag 
            : input.jump_mag / 2
        : 0;
    if (tick_slice == 0) output.next_state = input.is_wet ? SWIM_UP : IN_AIR;
    return output;
}

OutputState in_air_state(InputState input)
{
    OutputState output = init_output(IN_AIR);
    if (input.can_jump) output.next_state = input.motion_state.x > 200 
        ? LAND_HARD 
        : LAND_SOFT;
    if (input.motion_state.x > 50) output.next_state = input.is_wet ? SWIM_DOWN : FALLING_SLOW;
    return output;
}

OutputState swim_up_state(InputState input)
{
    OutputState output = init_output(SWIM_UP);
    if (input.can_jump) output.next_state = input.motion_state.x > 200 
        ? LAND_HARD 
        : LAND_SOFT;
    if (input.motion_state.x > 50) output.next_state = input.is_wet ? SWIM_DOWN : FALLING_SLOW;
    return output;
}

OutputState swim_down_state(InputState input)
{
    OutputState output = init_output(SWIM_DOWN);
    if (input.can_jump) output.next_state = input.motion_state.x > 200 
        ? LAND_HARD 
        : LAND_SOFT;
    if (input.motion_state.x > 200) output.next_state = input.is_wet ? SWIM_DOWN : FALLING_SLOW;
    if (input.motion_state.y > 50) output.next_state = input.is_wet ? SWIM_UP : IN_AIR;
    return output;
}

OutputState land_soft_state(InputState input)
{
    OutputState output = init_output(LAND_SOFT);
    if (input.current_time > 0.08f) output.next_state = IDLE;
    return output;
}

OutputState land_hard_state(InputState input)
{
    OutputState output = init_output(LAND_HARD);
    if (input.current_time > 0.22f) output.next_state = IDLE;
    return output;
}

OutputState punch_state(InputState input)
{
    OutputState output = init_output(PUNCH);
    if (!input.is_click_1) output.next_state = (input.is_mv_l || input.is_mv_r) ? WALKING : IDLE;
    if (input.can_jump && input.current_budget > 0 && input.mv_jump) output.next_state = RECOIL;
    if (output.next_state == PUNCH) output.attack = true;
    return output;
}

__kernel void set_control_points(__global int *control_flags,
                                 __global int *indices,
                                 __global float *move_mag,
                                 __global float *jump_mag,
                                 int target,
                                 int new_flags, 
                                 int new_index, 
                                 float new_move_mag, 
                                 float new_jump_mag)
{
    control_flags[target] = new_flags;
    indices[target] = new_index;
    move_mag[target] = new_move_mag;
    jump_mag[target] = new_jump_mag;
}

__kernel void handle_movement(__global float4 *entities,
                              __global float2 *entity_accel,
                              __global short2 *entity_motion_states,
                              __global int *entity_flags,
                              __global int2 *entity_animation_indices,
                              __global float2 *entity_animation_elapsed,
                              __global float2 *entity_animation_blend,
                              __global int *control_flags,
                              __global int *indices,
                              __global int *tick_budgets,
                              __global float *linear_mag,
                              __global float *jump_mag,
                              float dt)
{
    int current_control_set = get_global_id(0);

    int current_flags        = control_flags[current_control_set];
    int current_budget       = tick_budgets[current_control_set];
    float current_linear_mag = linear_mag[current_control_set];
    float current_jump_mag   = jump_mag[current_control_set];
    int current_index        = indices[current_control_set];

    float4 entity            = entities[current_index];
    float2 accel             = entity_accel[current_index];
    int arm_flag             = entity_flags[current_index];
    int2 anim_index          = entity_animation_indices[current_index];
    short2 motion_state      = entity_motion_states[current_index];
    float2 current_time      = entity_animation_elapsed[current_index];
    float2 current_blend     = entity_animation_blend[current_index];

    bool is_mv_l    = (current_flags & LEFT)   !=0;
    bool is_mv_r    = (current_flags & RIGHT)  !=0;
    bool is_mv_u    = (current_flags & UP)     !=0;
    bool is_mv_d    = (current_flags & DOWN)   !=0;
    bool is_click_1 = (current_flags & MOUSE1) !=0;
    bool is_click_2 = (current_flags & MOUSE2) !=0;
    bool mv_jump    = (current_flags & JUMP)   !=0;
    bool mv_run     = (current_flags & RUN)    !=0;
    bool can_jump   = (arm_flag & CAN_JUMP)    !=0;
    bool is_wet     = (arm_flag & IS_WET)      !=0;
    
    float move_mod = is_wet 
        ? 0.5f 
        : mv_run 
            ? 2.0f
            : 1.0f;

    current_budget = can_jump && !mv_jump
        ? 10 // todo: this should be a player stat, it is their jump height
        : current_budget;

    InputState input;
    input.is_mv_l        = is_mv_l;
    input.is_mv_r        = is_mv_r;
    input.mv_jump        = mv_jump;
    input.mv_run         = mv_run;
    input.can_jump       = can_jump;
    input.is_wet         = is_wet;
    input.is_click_1     = is_click_1;
    input.is_click_2     = is_click_2;
    input.current_budget = current_budget;
    input.motion_state   = motion_state;
    input.current_time   = current_time.x;
    input.anim_index     = anim_index.x;
    input.jump_mag       = current_jump_mag;

    OutputState state_result;
    switch(anim_index.x)
    {
        case IDLE:         state_result = idle_state(input);         break;
        case WALKING:      state_result = walking_state(input);      break;
        case RUNNING:      state_result = running_state(input);      break;
        case FALLING_SLOW: state_result = falling_slow_state(input); break;
        case FALLING_FAST: state_result = falling_fast_state(input); break;
        case RECOIL:       state_result = recoil_state(input);       break;
        case JUMPING:      state_result = jumping_state(input);      break;
        case IN_AIR:       state_result = in_air_state(input);       break;
        case SWIM_UP:      state_result = swim_up_state(input);      break;
        case SWIM_DOWN:    state_result = swim_down_state(input);    break;
        case LAND_SOFT:    state_result = land_soft_state(input);    break;
        case LAND_HARD:    state_result = land_hard_state(input);    break;
        case PUNCH:        state_result = punch_state(input);        break;
    }

    // transition handling

    bool blend = anim_index.x != state_result.next_state;

    current_time.y = blend 
        ? current_time.x
        : current_time.y;

    anim_index.y = blend 
        ? anim_index.x 
        : anim_index.y;

    current_blend = blend 
        ? (float2)(transition_table[anim_index.x][state_result.next_state], 0.0f)
        : current_blend;

    current_time.x = blend 
        ? 0.0f 
        : current_time.x;

    anim_index.x = state_result.next_state;

    // jumping

    accel.y = state_result.accel 
        ? state_result.jump_amount 
        : accel.y;

    current_budget = state_result.accel 
        ? state_result.next_budget
        : current_budget;

    // motion state

    float threshold = 10.0f;
    float2 vel = (entity.xy - entity.zw) / dt;

    motion_state.x = (vel.y < -threshold) 
        ? motion_state.x + 1 
        : 0;

    motion_state.y = (vel.y > threshold) 
        ? motion_state.y + 1 
        : 0;

    motion_state.x = motion_state.x > 1000 
        ? 1000 
        : motion_state.x;

    motion_state.y = motion_state.y > 1000 
        ? 1000 
        : motion_state.y;

    // acceleration

    accel.x = is_mv_l && !is_mv_r
        ? -current_linear_mag * move_mod
        : accel.x;

    accel.x = is_mv_r && !is_mv_l
        ? current_linear_mag * move_mod
        : accel.x;

    accel.x = !is_mv_r && !is_mv_l
        ? 0.0f
        : accel.x;

    accel.y = is_mv_u && is_wet
        ? current_linear_mag * 1.5
        : accel.y;

    accel.y = is_mv_d && is_wet
        ? -current_linear_mag
        : accel.y;

    // set flags

    arm_flag = is_mv_l != is_mv_r 
        ? is_mv_l
            ? arm_flag | FACE_LEFT 
            : is_mv_r 
                ? arm_flag & ~FACE_LEFT 
                : arm_flag
        : arm_flag; 

    arm_flag = state_result.attack 
        ? arm_flag | ATTACKING 
        : arm_flag & ~ATTACKING; 

    arm_flag = is_click_2 
        ? arm_flag | CAN_COLLECT 
        : arm_flag & ~CAN_COLLECT; 

    //printf("debug: current_blend= %f,%f\n", current_blend.x, current_blend.y);

    entity_accel[current_index]             = accel;
    entity_flags[current_index]             = arm_flag;
    tick_budgets[current_control_set]       = current_budget;
    entity_motion_states[current_index]     = motion_state;
    entity_animation_blend[current_index]   = current_blend;
    entity_animation_elapsed[current_index] = current_time;
    entity_animation_indices[current_index] = anim_index;
}