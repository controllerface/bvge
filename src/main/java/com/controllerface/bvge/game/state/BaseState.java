package com.controllerface.bvge.game.state;

import com.controllerface.bvge.game.PlayerInput;

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
