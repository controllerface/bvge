package com.controllerface.bvge.game.state;

import com.controllerface.bvge.ecs.components.PlayerInput;
import com.controllerface.bvge.physics.StateInput;
import com.controllerface.bvge.physics.StateOutput;

import static com.controllerface.bvge.ecs.components.InputBinding.*;

public enum ActionState
{
    NONE    (AnimationState.IDLE),
    PUNCH   (AnimationState.PUNCH),

    ;

    public final AnimationState animation;

    ActionState(AnimationState animation)
    {
        this.animation = animation;
    }

    public static ActionState process(StateInput input,
                                      StateOutput output,
                                      ActionState current_state,
                                      PlayerInput player)
    {
        return switch (current_state)
        {
            case NONE ->
            {
                var state = NONE;
                if (player.pressed(MOUSE_PRIMARY) && input.can_click) state = PUNCH;
                if (state == PUNCH) output.attack = true;
                yield state;
            }

            case PUNCH ->
            {
                var state = PUNCH;
                if (!player.pressed(MOUSE_PRIMARY)) state = NONE;
                if (state == PUNCH) output.attack = true;
                yield state;
            }
        };
    }
}
