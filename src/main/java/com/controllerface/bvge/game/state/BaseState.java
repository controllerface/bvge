package com.controllerface.bvge.game.state;

import com.controllerface.bvge.ecs.components.PlayerInput;
import com.controllerface.bvge.physics.StateInput;
import com.controllerface.bvge.physics.StateOutput;

public enum BaseState
{
    IDLE    (AnimationState.IDLE),

    ;

    public final AnimationState animation;

    BaseState(AnimationState animation)
    {
        this.animation = animation;
    }

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
