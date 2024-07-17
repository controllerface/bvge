package com.controllerface.bvge.game.state;

import com.controllerface.bvge.ecs.components.PlayerInput;
import com.controllerface.bvge.physics.StateInput;
import com.controllerface.bvge.physics.StateOutput;

import static com.controllerface.bvge.ecs.components.InputBinding.*;

public enum MovementState
{
    REST            (AnimationState.IDLE),
    WALKING         (AnimationState.WALKING),
    RUNNING         (AnimationState.RUNNING),
    FALLING_FAST    (AnimationState.FALLING_FAST),
    RECOIL          (AnimationState.RECOIL),
    JUMPING         (AnimationState.JUMPING),
    IN_AIR          (AnimationState.IN_AIR),
    LAND_HARD       (AnimationState.LAND_HARD),
    FALLING_SLOW    (AnimationState.FALLING_SLOW),
    LAND_SOFT       (AnimationState.LAND_SOFT),
    SWIM_UP         (AnimationState.SWIM_UP),
    SWIM_DOWN       (AnimationState.SWIM_DOWN),

    ;

    public final AnimationState animation;

    MovementState(AnimationState animation)
    {
        this.animation = animation;
    }

    private static void init_output(StateOutput output)
    {
        output.jumping     = false;
        output.attack      = false;
        output.next_budget = 0;
        output.jump_amount = 0.0f;
    }

    public static MovementState process(StateInput input,
                                        StateOutput output,
                                        MovementState current_state,
                                        PlayerInput player)
    {

        init_output(output);
        var state = current_state;
        return switch (current_state)
        {
            case REST ->
            {
                if (player.pressed(MOVE_LEFT) || player.pressed(MOVE_RIGHT))
                {
                    state = player.pressed(RUN)
                        ? RUNNING
                        : WALKING;
                }
                if (input.can_jump && input.current_budget > 0 && player.pressed(JUMP))
                {
                    state = RECOIL;
                }
                if (input.motion_state[0] > 100)
                {
                    state = input.is_wet
                        ? SWIM_DOWN
                        : FALLING_SLOW;
                }
                if (input.motion_state[1] > 150)
                {
                    state = input.is_wet
                        ? SWIM_UP
                        : IN_AIR;
                }
                yield state;
            }

            case WALKING ->
            {
                if (player.pressed(RUN))
                {
                    state = RUNNING;
                }
                if (!player.pressed(MOVE_LEFT) && !player.pressed(MOVE_RIGHT))
                {
                    state = REST;
                }
                if (input.can_jump && input.current_budget > 0 && player.pressed(JUMP))
                {
                    state = RECOIL;
                }
                if (input.motion_state[0] > 100)
                {
                    state = input.is_wet
                        ? SWIM_DOWN
                        : FALLING_SLOW;
                }
                if (input.motion_state[1] > 150)
                {
                    state = input.is_wet
                        ? SWIM_UP
                        : IN_AIR;
                }
                yield state;
            }

            case RUNNING ->
            {
                if (!player.pressed(RUN))
                {
                    state = WALKING;
                }
                if (!player.pressed(MOVE_LEFT) && !player.pressed(MOVE_RIGHT))
                {
                    state = REST;
                }
                if (input.can_jump && input.current_budget > 0 && player.pressed(JUMP))
                {
                    state = RECOIL;
                }
                if (input.motion_state[0] > 100)
                {
                    state = input.is_wet
                        ? SWIM_DOWN
                        : FALLING_SLOW;
                }
                if (input.motion_state[1] > 150)
                {
                    state = input.is_wet
                        ? SWIM_UP
                        : IN_AIR;
                }
                yield state;
            }

            case FALLING_FAST ->
            {
                if (input.can_jump)
                {
                    state = input.motion_state[0] > 200
                        ? LAND_HARD
                        : LAND_SOFT;
                }
                if (input.is_wet)
                {
                    state = SWIM_DOWN;
                }
                if (input.motion_state[0] < 200)
                {
                    state = input.is_wet
                        ? SWIM_DOWN
                        : FALLING_SLOW;
                }
                if (input.motion_state[1] > 50)
                {
                    state = input.is_wet
                        ? SWIM_UP
                        : IN_AIR;
                }
                yield state;
            }

            case RECOIL ->
            {
                if (input.current_time > 0.15f)
                {
                    state = JUMPING;
                }
                yield state;
            }

            case JUMPING ->
            {
                output.jumping = true;
                output.next_budget = input.current_budget;
                int jump_cost = input.current_budget > 0
                    ? 1
                    : 0;
                output.next_budget -= jump_cost;
                output.jump_amount = jump_cost == 1
                    ? player.pressed(JUMP)
                        ? input.jump_mag
                        : input.jump_mag / 2
                    : 0;
                if (jump_cost == 0)
                {
                    state = input.is_wet
                        ? SWIM_UP
                        : IN_AIR;
                }
                yield state;
            }

            case IN_AIR ->
            {
                if (input.can_jump)
                {
                    state = input.motion_state[0] > 200
                        ? LAND_HARD
                        : LAND_SOFT;
                }
                if (input.motion_state[0] > 100)
                {
                    state = input.is_wet
                        ? SWIM_DOWN
                        : FALLING_SLOW;
                }
                yield state;
            }

            case LAND_HARD ->
            {
                if (input.current_time > 0.22f)
                {
                    if (player.pressed(MOVE_LEFT) || player.pressed(MOVE_RIGHT))
                    {
                        state = player.pressed(RUN)
                            ? RUNNING
                            : WALKING;
                    }
                    else state = REST;
                }
                yield state;
            }

            case FALLING_SLOW ->
            {
                if (input.can_jump)
                {
                    state = input.motion_state[0] > 200
                        ? LAND_HARD
                        : LAND_SOFT;
                }
                if (input.motion_state[0] > 200)
                {
                    state = input.is_wet
                        ? SWIM_DOWN
                        : FALLING_FAST;
                }
                if (input.motion_state[1] > 50)
                {
                    state = input.is_wet
                        ? SWIM_UP
                        : IN_AIR;
                }
                yield state;
            }

            case LAND_SOFT ->
            {
                if (input.current_time > 0.40f)
                {
                    if (player.pressed(MOVE_LEFT) || player.pressed(MOVE_RIGHT))
                    {
                        state = player.pressed(RUN)
                            ? RUNNING
                            : WALKING;
                    }
                    else state = REST;
                }
                yield state;
            }

            case SWIM_UP ->
            {
                if (input.can_jump)
                {
                    state = input.motion_state[0] > 200
                        ? LAND_HARD
                        : LAND_SOFT;
                }
                if (input.motion_state[0] > 50)
                {
                    state = input.is_wet
                        ? SWIM_DOWN
                        : FALLING_SLOW;
                }
                yield state;
            }

            case SWIM_DOWN ->
            {
                if (input.can_jump)
                {
                    state = input.motion_state[0] > 200
                        ? LAND_HARD
                        : LAND_SOFT;
                }
                if (input.motion_state[0] > 200)
                {
                    state = input.is_wet
                        ? SWIM_DOWN
                        : FALLING_SLOW;
                }
                if (input.motion_state[1] > 50)
                {
                    state = input.is_wet
                        ? SWIM_UP
                        : IN_AIR;
                }
                yield state;
            }
        };
    }
}
