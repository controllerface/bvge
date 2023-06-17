package com.controllerface.bvge.ecs;

import org.joml.Vector2f;

public class ControlPoints implements GameComponent
{
    private boolean disabled = false;

    // cardinal directions
    private boolean up = false;
    private boolean down = false;
    private boolean left = false;
    private boolean right = false;

    // mouse buttons
    private boolean primary = false;
    private boolean secondary = false;
    private boolean middle = false;
    private boolean back = false;
    private boolean forward = false;

    // target position
    private Vector2f target = new Vector2f();

    public boolean isDisabled()
    {
        return disabled;
    }

    public void setDisabled(boolean disabled)
    {
        this.disabled = disabled;
    }

    public boolean isUp()
    {
        return up;
    }

    public void setUp(boolean up)
    {
        this.up = up;
    }

    public boolean isDown()
    {
        return down;
    }

    public void setDown(boolean down)
    {
        this.down = down;
    }

    public boolean isLeft()
    {
        return left;
    }

    public void setLeft(boolean left)
    {
        this.left = left;
    }

    public boolean isRight()
    {
        return right;
    }

    public void setRight(boolean right)
    {
        this.right = right;
    }

    public boolean isPrimary()
    {
        return primary;
    }

    public void setPrimary(boolean primary)
    {
        this.primary = primary;
    }

    public boolean isSecondary()
    {
        return secondary;
    }

    public void setSecondary(boolean secondary)
    {
        this.secondary = secondary;
    }

    public boolean isMiddle()
    {
        return middle;
    }

    public void setMiddle(boolean middle)
    {
        this.middle = middle;
    }

    public boolean isBack()
    {
        return back;
    }

    public void setBack(boolean back)
    {
        this.back = back;
    }

    public boolean isForward()
    {
        return forward;
    }

    public void setForward(boolean forward)
    {
        this.forward = forward;
    }

    public Vector2f getTarget()
    {
        return target;
    }
}
