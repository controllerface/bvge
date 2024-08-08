package com.controllerface.bvge.game;

public enum InputBinding
{
    MOVE_UP(),
    MOVE_DOWN(),
    MOVE_LEFT(),
    MOVE_RIGHT(),

    JUMP(),
    RUN(),

    MOUSE_PRIMARY(true),
    MOUSE_SECONDARY(true),
    MOUSE_MIDDLE(true),
    MOUSE_BACK(true),
    MOUSE_FORWARD(true),

    ;

    public final boolean mouse;

    InputBinding(boolean mouse)
    {
        this.mouse = mouse;
    }

    InputBinding()
    {
        this(false);
    }
}
