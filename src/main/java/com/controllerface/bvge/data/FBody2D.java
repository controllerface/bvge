package com.controllerface.bvge.data;

import com.controllerface.bvge.Main;
import com.controllerface.bvge.ecs.components.GameComponent;
import org.joml.Vector2f;

public class FBody2D implements GameComponent
{
    int index;
    float force;
    float x_pos;
    float y_pos;

    public FBody2D(int index, float force)
    {
        this.index = index;
        this.force = force;
    }

    public int index()
    {
        return index;
    }

    public float force()
    {
        return force;
    }

    public float x_pos()
    {
        return x_pos;
    }

    public float y_pos()
    {
        return y_pos;
    }

    public void set_x_pos(float x_pos)
    {
        this.x_pos = x_pos;
    }

    public void set_y_pos(float y_pos)
    {
        this.y_pos = y_pos;
    }

    /*
     * Memory layout: float16
     *  0: x position                 (transform)
     *  1: y position                 (transform)
     *  2: scale x                    (transform)
     *  3: scale y                    (transform)
     *  4: acceleration x component
     *  5: acceleration y component
     *  6: collision flags            (int cast)
     *  7: start point index          (int cast)
     *  8: end point index            (int cast)
     *  9: start edge index           (int cast)
     * 10: end edge index             (int cast)
     * 11:
     * 12:
     * 13:
     * 14:
     * 15:
     *  */
    public int bodyIndex() { return index() / Main.Memory.Width.BODY; }
}
