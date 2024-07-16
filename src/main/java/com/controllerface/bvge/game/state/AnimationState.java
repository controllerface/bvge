package com.controllerface.bvge.game.state;

import com.controllerface.bvge.ecs.components.PlayerInput;
import com.controllerface.bvge.physics.StateInput;
import com.controllerface.bvge.physics.StateOutput;

import java.util.Arrays;

import static com.controllerface.bvge.ecs.components.InputBinding.*;

public enum AnimationState
{
    IDLE            (0),
    WALKING         (1),
    RUNNING         (1),
    FALLING_FAST    (1),
    RECOIL          (1),
    JUMPING         (1),
    IN_AIR          (1),
    LAND_HARD       (1),
    FALLING_SLOW    (1),
    LAND_SOFT       (1),
    SWIM_UP         (1),
    SWIM_DOWN       (1),
    PUNCH           (2),
    UNKNOWN         (1),

    ;

    public final int layer;

    AnimationState(int layer)
    {
        this.layer = layer;
    }


    public static AnimationState from_index(int index)
    {
        return (index < 0 || index >= AnimationState.values().length)
            ? UNKNOWN
            : AnimationState.values()[index];
    }

    public static AnimationState fuzzy_match(String animation_name)
    {
        if (animation_name == null) return UNKNOWN;

        return Arrays.stream(values())
            .filter(state -> animation_name.toUpperCase().contains(state.name()))
            .findAny().orElse(UNKNOWN);
    }

    public static float blend_time(AnimationState from, AnimationState to)
    {
        return switch (from)
        {
            case IDLE -> switch (to)
            {
                case IDLE, UNKNOWN -> 0.0f;
                case WALKING,
                     LAND_HARD,
                     FALLING_SLOW,
                     LAND_SOFT,
                     SWIM_UP,
                     SWIM_DOWN,
                     FALLING_FAST,
                     JUMPING,
                     RUNNING,
                     IN_AIR -> 0.4f;
                case PUNCH,
                     RECOIL -> 0.1f;
            };

            case WALKING -> switch (to)
            {
                case WALKING, UNKNOWN -> 0.0f;
                case IDLE,
                     LAND_HARD,
                     FALLING_SLOW,
                     LAND_SOFT,
                     SWIM_UP,
                     SWIM_DOWN,
                     FALLING_FAST,
                     JUMPING,
                     RUNNING,
                     IN_AIR -> 0.2f;
                case PUNCH,
                     RECOIL -> 0.1f;
            };

            case FALLING_FAST -> switch (to)
            {
                case FALLING_FAST, UNKNOWN -> 0.0f;
                case WALKING,
                     LAND_HARD,
                     FALLING_SLOW,
                     LAND_SOFT,
                     SWIM_UP,
                     SWIM_DOWN,
                     IDLE,
                     JUMPING,
                     RUNNING,
                     IN_AIR -> 0.2f;
                case PUNCH,
                     RECOIL -> 0.1f;
            };

            case IN_AIR -> switch (to)
            {
                case IN_AIR, UNKNOWN -> 0.0f;
                case WALKING,
                     FALLING_SLOW,
                     FALLING_FAST,
                     LAND_SOFT,
                     SWIM_UP,
                     SWIM_DOWN,
                     IDLE,
                     JUMPING,
                     RUNNING,
                     LAND_HARD -> 0.2f;
                case PUNCH,
                     RECOIL -> 0.1f;
            };

            case FALLING_SLOW -> switch (to)
            {
                case FALLING_SLOW, UNKNOWN -> 0.0f;
                case WALKING,
                     LAND_HARD,
                     FALLING_FAST,
                     LAND_SOFT,
                     SWIM_UP,
                     SWIM_DOWN,
                     IDLE,
                     JUMPING,
                     RUNNING,
                     IN_AIR -> 0.2f;
                case PUNCH,
                     RECOIL -> 0.1f;
            };

            case SWIM_UP -> switch (to)
            {
                case SWIM_UP, UNKNOWN -> 0.0f;
                case WALKING,
                     LAND_HARD,
                     FALLING_SLOW,
                     LAND_SOFT,
                     SWIM_DOWN,
                     FALLING_FAST,
                     IDLE,
                     JUMPING,
                     RUNNING,
                     IN_AIR -> 0.2f;
                case PUNCH,
                     RECOIL -> 0.1f;
            };

            case SWIM_DOWN -> switch (to)
            {
                case SWIM_DOWN, UNKNOWN -> 0.0f;
                case WALKING,
                     LAND_HARD,
                     FALLING_SLOW,
                     LAND_SOFT,
                     SWIM_UP,
                     FALLING_FAST,
                     IDLE,
                     JUMPING,
                     RUNNING,
                     IN_AIR -> 0.2f;
                case PUNCH,
                     RECOIL -> 0.1f;
            };

            case PUNCH -> switch (to)
            {
                case PUNCH, UNKNOWN -> 0.0f;
                case IDLE,
                     WALKING,
                     JUMPING -> 0.3f;
                case RUNNING,
                     RECOIL,
                     IN_AIR,
                     LAND_HARD,
                     LAND_SOFT,
                     SWIM_UP,
                     SWIM_DOWN -> 0.2f;
                case FALLING_FAST,
                     FALLING_SLOW -> 0.5f;
            };

            case RUNNING -> switch (to)
            {
                case RUNNING, UNKNOWN -> 0.0f;
                case LAND_HARD,
                     FALLING_SLOW,
                     LAND_SOFT,
                     SWIM_UP,
                     SWIM_DOWN,
                     FALLING_FAST,
                     JUMPING,
                     IN_AIR -> 0.2f;
                case IDLE,
                     WALKING -> 0.4f;
                case PUNCH, RECOIL -> 0.1f;
            };

            case JUMPING -> 0.5f;
            case RECOIL, LAND_HARD, LAND_SOFT -> 0.05f;
            case UNKNOWN -> 0.0f;
        };
    }

    private static void init_output(StateOutput output)
    {
        output.jumping     = false;
        output.attack      = false;
        output.next_budget = 0;
        output.jump_amount = 0.0f;
    }

    private static AnimationState state = UNKNOWN;

    public static AnimationState process(StateInput input,
                                         StateOutput output,
                                         AnimationState current_state,
                                         PlayerInput player)
    {
        init_output(output);
        state = current_state;
        return switch (current_state)
        {
            case UNKNOWN -> state;

            case IDLE ->
            {
                if (player.pressed(MOVE_LEFT) || player.pressed(MOVE_RIGHT)) state = player.pressed(RUN) ? RUNNING : WALKING;
                if (player.pressed(MOUSE_PRIMARY) && input.can_click) state = PUNCH;
                if (input.can_jump && input.current_budget > 0 && player.pressed(JUMP)) state = RECOIL;
                if (input.motion_state[0] > 100) state = input.is_wet ? SWIM_DOWN : FALLING_SLOW;
                if (input.motion_state[1] > 150) state = input.is_wet ? SWIM_UP : IN_AIR;
                yield state;
            }

            case WALKING ->
            {
                if (player.pressed(RUN)) state = RUNNING;
                if (!player.pressed(MOVE_LEFT) && !player.pressed(MOVE_RIGHT)) state = IDLE;
                if (player.pressed(MOUSE_PRIMARY) && input.can_click) state = PUNCH;
                if (input.can_jump && input.current_budget > 0 && player.pressed(JUMP)) state = RECOIL;
                if (input.motion_state[0] > 100) state = input.is_wet ? SWIM_DOWN : FALLING_SLOW;
                if (input.motion_state[1] > 150) state = input.is_wet ? SWIM_UP : IN_AIR;
                yield state;
            }

            case RUNNING ->
            {
                if (!player.pressed(RUN)) state = WALKING;
                if (!player.pressed(MOVE_LEFT) && !player.pressed(MOVE_RIGHT)) state = IDLE;
                if (player.pressed(MOUSE_PRIMARY) && input.can_click) state = PUNCH;
                if (input.can_jump && input.current_budget > 0 && player.pressed(JUMP)) state = RECOIL;
                if (input.motion_state[0] > 100) state = input.is_wet ? SWIM_DOWN : FALLING_SLOW;
                if (input.motion_state[1] > 150) state = input.is_wet ? SWIM_UP : IN_AIR;
                yield state;
            }

            case FALLING_FAST ->
            {
                if (input.can_jump) state = input.motion_state[0] > 200 ? LAND_HARD : LAND_SOFT;
                if (input.is_wet) state = SWIM_DOWN;
                if (input.motion_state[0] < 200) state = input.is_wet ? SWIM_DOWN : FALLING_SLOW;
                if (input.motion_state[1] > 50) state = input.is_wet ? SWIM_UP : IN_AIR;
                yield state;
            }

            case RECOIL ->
            {
                if (input.current_time > 0.15f) state = JUMPING;
                yield state;
            }

            case JUMPING ->
            {
                output.jumping = true;
                output.next_budget = input.current_budget;
                int jump_cost = input.current_budget > 0 ? 1 : 0;
                output.next_budget -= jump_cost;
                output.jump_amount = jump_cost == 1
                    ? player.pressed(JUMP)
                        ? input.jump_mag
                        : input.jump_mag / 2
                    : 0;
                if (jump_cost == 0) state = input.is_wet ? SWIM_UP : IN_AIR;
                yield state;
            }

            case IN_AIR ->
            {
                if (input.can_jump) state = input.motion_state[0] > 200 ? LAND_HARD : LAND_SOFT;
                if (input.motion_state[0] > 100) state = input.is_wet ? SWIM_DOWN : FALLING_SLOW;
                yield state;
            }

            case LAND_HARD ->
            {
                if (input.current_time > 0.22f) state = IDLE;
                yield state;
            }

            case FALLING_SLOW ->
            {
                if (input.can_jump) state = input.motion_state[0] > 200 ? LAND_HARD : LAND_SOFT;
                if (input.motion_state[0] > 200) state = input.is_wet ? SWIM_DOWN : FALLING_FAST;
                if (input.motion_state[1] > 50) state = input.is_wet ? SWIM_UP : IN_AIR;
                yield state;
            }

            case LAND_SOFT ->
            {
                if (input.current_time > 0.08f) state = IDLE;
                yield state;
            }

            case SWIM_UP ->
            {
                if (input.can_jump) state = input.motion_state[0] > 200 ? LAND_HARD : LAND_SOFT;
                if (input.motion_state[0] > 50) state = input.is_wet ? SWIM_DOWN : FALLING_SLOW;
                yield state;
            }

            case SWIM_DOWN ->
            {
                if (input.can_jump) state = input.motion_state[0] > 200 ? LAND_HARD : LAND_SOFT;
                if (input.motion_state[0] > 200) state = input.is_wet ? SWIM_DOWN : FALLING_SLOW;
                if (input.motion_state[1] > 50) state = input.is_wet ? SWIM_UP : IN_AIR;
                yield state;
            }

            case PUNCH ->
            {
                if (!player.pressed(MOUSE_PRIMARY)) state = (player.pressed(MOVE_LEFT) || player.pressed(MOVE_RIGHT)) ? WALKING : IDLE;
                if (player.pressed(JUMP) && input.can_jump && input.current_budget > 0) state = RECOIL;
                if (state == PUNCH) output.attack = true;
                yield state;
            }
        };
    }
}