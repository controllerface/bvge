package com.controllerface.bvge.data;

import com.controllerface.bvge.Main;
import com.controllerface.bvge.ecs.components.GameComponent;
import org.joml.Vector2f;

public record FBody2D(int index,
                      float force,
                      FPoint2D[] points,
                      FEdge2D[] edges,
                      FBounds2D bounds,
                      FTransform transform,
                      String entity) implements GameComponent
{
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
    public static final int X_OFFSET            = 0;
    public static final int Y_OFFSET            = 1;
    public static final int SX_OFFSET           = 2;
    public static final int SY_OFFSET           = 3;
    public static final int ACC_X_OFFSET        = 4;
    public static final int ACC_Y_OFFSET        = 5;
    public static final int FLAGS_OFFSET        = 6;
    public static final int SP_OFFSET           = 7;
    public static final int EP_OFFSET           = 8;
    public static final int SE_OFFSET           = 9;
    public static final int EE_OFFSET           = 10;

    public int bodyIndex() { return index() / Main.Memory.Width.BODY; }

    public float pos_x()
    {
        return Main.Memory.body_buffer[index() + X_OFFSET];
    }

    public float pos_y()
    {
        return Main.Memory.body_buffer[index() + Y_OFFSET];
    }

    public float scale_x()
    {
        return Main.Memory.body_buffer[index() + SX_OFFSET];
    }

    public float scale_y()
    {
        return Main.Memory.body_buffer[index() + SY_OFFSET];
    }

    public float acc_x()
    {
        return Main.Memory.body_buffer[index() + ACC_X_OFFSET];
    }

    public float acc_y()
    {
        return Main.Memory.body_buffer[index() + ACC_Y_OFFSET];
    }

    public int flags()
    {
        return (int)Main.Memory.body_buffer[index() + FLAGS_OFFSET];
    }

    public int start_point()
    {
        return (int)Main.Memory.body_buffer[index() + SP_OFFSET];
    }

    public int end_point()
    {
        return (int)Main.Memory.body_buffer[index() + EP_OFFSET];
    }

    public int start_edge()
    {
        return (int)Main.Memory.body_buffer[index() + SE_OFFSET];
    }

    public int end_edge()
    {
        return (int)Main.Memory.body_buffer[index() + EE_OFFSET];
    }

    public void setPos(Vector2f newPos)
    {
        Main.Memory.body_buffer[index() + X_OFFSET] = newPos.x;
        Main.Memory.body_buffer[index() + Y_OFFSET] = newPos.y;
    }

    public void setAcc(Vector2f newAcc)
    {
        Main.Memory.body_buffer[index() + ACC_X_OFFSET] = newAcc.x;
        Main.Memory.body_buffer[index() + ACC_Y_OFFSET] = newAcc.y;
    }

    public void addAcc(Vector2f newAcc)
    {
        Main.Memory.body_buffer[index() + ACC_X_OFFSET] += newAcc.x;
        Main.Memory.body_buffer[index() + ACC_Y_OFFSET] += newAcc.y;
    }
}
