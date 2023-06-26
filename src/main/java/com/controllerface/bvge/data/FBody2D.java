package com.controllerface.bvge.data;

import com.controllerface.bvge.Main;
import com.controllerface.bvge.ecs.components.GameComponent;
import org.joml.Vector2f;

public record FBody2D(int index, float force,
                      FPoint2D[] points, FEdge2D[] edges,
                      FBounds2D bounds, FTransform transform,
                      String entity) implements GameComponent
{
    /*
     * Memory layout: float16
     *  0: x position (transform)
     *  1: y position (transform)
     *  2: scale x    (transform)
     *  3: scale y    (transform)
     *  4: acceleration x component
     *  5: acceleration y component
     *  6: bounding box index (int cast)
     *  7: start point index  (int cast)
     *  8: end point index    (int cast)
     *  9: start edge index   (int cast)
     * 10: end edge index     (int cast)
     * 11: [empty]
     * 12: [empty]
     * 13: [empty]
     * 14: [empty]
     * 15: [empty]
     *  */
    public static int x_offset = 0;
    public static int y_offset = 1;
    public static int sx_offset = 2;
    public static int sy_offset = 3;
    public static int acc_x_offset = 4;
    public static int acc_y_offset = 5;
    public static int bi_offset = 6;
    public static int sp_offset = 7;
    public static int ep_offset = 8;
    public static int se_offset = 9;
    public static int ee_offset = 10;

    public float pos_x()
    {
        return Main.Memory.body_buffer[index() + x_offset];
    }

    public float pos_y()
    {
        return Main.Memory.body_buffer[index() + y_offset];
    }

    public float scale_x()
    {
        return Main.Memory.body_buffer[index() + sx_offset];
    }

    public float scale_y()
    {
        return Main.Memory.body_buffer[index() + sy_offset];
    }

    public float acc_x()
    {
        return Main.Memory.body_buffer[index() + acc_x_offset];
    }

    public float acc_y()
    {
        return Main.Memory.body_buffer[index() + acc_y_offset];
    }

    public float bounds_i()
    {
        return Main.Memory.body_buffer[index() + bi_offset];
    }

    public int start_point()
    {
        return (int)Main.Memory.body_buffer[index() + sp_offset];
    }

    public int end_point()
    {
        return (int)Main.Memory.body_buffer[index() + ep_offset];
    }

    public int start_edge()
    {
        return (int)Main.Memory.body_buffer[index() + se_offset];
    }

    public int end_edge()
    {
        return (int)Main.Memory.body_buffer[index() + ee_offset];
    }

    public void setPos(Vector2f newPos)
    {
        Main.Memory.body_buffer[index() + x_offset] = newPos.x;
        Main.Memory.body_buffer[index() + y_offset] = newPos.y;
    }

    public void setAcc(Vector2f newAcc)
    {
        Main.Memory.body_buffer[index() + acc_x_offset] = newAcc.x;
        Main.Memory.body_buffer[index() + acc_y_offset] = newAcc.y;
    }

    public void mulAcc(float scalar)
    {
        Main.Memory.body_buffer[index() + acc_x_offset] = acc_x() * scalar;
        Main.Memory.body_buffer[index() + acc_y_offset] = acc_y() * scalar;
    }
}
