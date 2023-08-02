package com.controllerface.bvge.ecs.components;

import org.joml.Vector2f;

public class ControlPoints implements GameComponent
{
    private boolean disabled = false;

    // cardinal directions
    private boolean up = false;
    private boolean down = false;
    private boolean left = false;
    private boolean right = false;
    private boolean rot_left = false;
    private boolean rot_right = false;

    // mouse buttons
    private boolean primary = false;
    private boolean secondary = false;
    private boolean middle = false;
    private boolean back = false;
    private boolean forward = false;

    // keyboard
    private boolean space_bar = false;

    // target position
    private Vector2f target = new Vector2f();

    public boolean is_disabled()
    {
        return disabled;
    }

    public void set_disabled(boolean disabled)
    {
        this.disabled = disabled;
    }

    public boolean is_moving_up()
    {
        return up;
    }

    public void set_moving_up(boolean up)
    {
        this.up = up;
    }

    public boolean is_moving_down()
    {
        return down;
    }

    public void set_moving_down(boolean down)
    {
        this.down = down;
    }

    public boolean is_moving_left()
    {
        return left;
    }

    public void set_moving_Left(boolean left)
    {
        this.left = left;
    }

    public boolean is_moving_right()
    {
        return right;
    }

    public void set_moving_right(boolean right)
    {
        this.right = right;
    }

    public boolean is_rotating_left()
    {
        return rot_left;
    }

    public void set_rotating_Left(boolean left)
    {
        this.rot_left = left;
    }

    public boolean is_rotating_right()
    {
        return rot_right;
    }

    public void set_rot_right(boolean right)
    {
        this.rot_right = right;
    }

    public boolean is_space_bar_down()
    {
        return space_bar;
    }

    public void set_space_bar(boolean space_bar)
    {
        this.space_bar = space_bar;
    }

    public boolean is_primary()
    {
        return primary;
    }

    public void set_primary(boolean primary)
    {
        this.primary = primary;
    }

    public boolean is_secondary()
    {
        return secondary;
    }

    public void set_secondary(boolean secondary)
    {
        this.secondary = secondary;
    }

    public boolean is_middle()
    {
        return middle;
    }

    public void set_middle(boolean middle)
    {
        this.middle = middle;
    }

    public boolean is_back()
    {
        return back;
    }

    public void set_back(boolean back)
    {
        this.back = back;
    }

    public boolean is_forward()
    {
        return forward;
    }

    public void set_forward(boolean forward)
    {
        this.forward = forward;
    }

    public Vector2f get_target()
    {
        return target;
    }
}
