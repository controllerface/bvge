package com.controllerface.bvge.game.state;

import com.controllerface.bvge.ecs.components.PlayerInput;
import com.controllerface.bvge.physics.StateInput;
import com.controllerface.bvge.physics.StateOutput;

import static com.controllerface.bvge.ecs.components.InputBinding.*;
import static com.controllerface.bvge.ecs.components.InputBinding.JUMP;

public class PlayerState
{
    private static BaseState base_state     = BaseState.IDLE;
    private static ActionState action_state = ActionState.IDLE;
    private static MovementState move_state = MovementState.IDLE;

    private static void init_output(StateOutput output)
    {
        output.jumping     = false;
        output.attack      = false;
        output.next_budget = 0;
        output.jump_amount = 0.0f;
    }

    public enum BaseState
    {
        IDLE,

        ;

        public static BaseState process(StateInput input,
                          StateOutput output,
                          BaseState current_state,
                          PlayerInput player)
        {
            return switch (current_state)
            {
                case IDLE -> IDLE;
            };
        }
    }

    public enum MovementState
    {
        IDLE,
        WALKING,
        RUNNING,
        FALLING_FAST,
        RECOIL,
        JUMPING,
        IN_AIR,
        LAND_HARD,
        FALLING_SLOW,
        LAND_SOFT,
        SWIM_UP,
        SWIM_DOWN,

        ;

        public static MovementState process(StateInput input,
                              StateOutput output,
                              MovementState current_state,
                              PlayerInput player)
        {

            init_output(output);
            var state = current_state;
            return switch (current_state)
            {
                case IDLE ->
                {
                    if (player.pressed(MOVE_LEFT) || player.pressed(MOVE_RIGHT)) state = player.pressed(RUN) ? RUNNING : WALKING;
                    if (input.can_jump && input.current_budget > 0 && player.pressed(JUMP)) state = RECOIL;
                    if (input.motion_state[0] > 100) state = input.is_wet ? SWIM_DOWN : FALLING_SLOW;
                    if (input.motion_state[1] > 150) state = input.is_wet ? SWIM_UP : IN_AIR;
                    yield state;
                }

                case WALKING ->
                {
                    if (player.pressed(RUN)) state = RUNNING;
                    if (!player.pressed(MOVE_LEFT) && !player.pressed(MOVE_RIGHT)) state = IDLE;
                    if (input.can_jump && input.current_budget > 0 && player.pressed(JUMP)) state = RECOIL;
                    if (input.motion_state[0] > 100) state = input.is_wet ? SWIM_DOWN : FALLING_SLOW;
                    if (input.motion_state[1] > 150) state = input.is_wet ? SWIM_UP : IN_AIR;
                    yield state;
                }

                case RUNNING ->
                {
                    if (!player.pressed(RUN)) state = WALKING;
                    if (!player.pressed(MOVE_LEFT) && !player.pressed(MOVE_RIGHT)) state = IDLE;
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
            };
        }
    }

    public enum ActionState
    {
        IDLE,
        PUNCH,

        ;

        public static ActionState process(StateInput input,
                StateOutput output,
                ActionState current_state,
                PlayerInput player)
        {
            return switch (current_state)
            {
                case IDLE -> player.pressed(MOUSE_PRIMARY) && input.can_click
                    ? PUNCH
                    : IDLE;

                case PUNCH -> player.pressed(MOUSE_PRIMARY)
                    ? PUNCH
                    : IDLE;
            };
        }
    }
}
