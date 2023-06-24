package com.controllerface.bvge.data;

import com.controllerface.bvge.Main;
import com.controllerface.bvge.ecs.components.GameComponent;
import org.joml.Vector2f;

public record FBody2D(int index, float force,
                      FPoint2D[] points, FEdge2D[] edges,
                      String entity) implements GameComponent
{

    private static int x_offset = 0;
    private static int y_offset = 1;
    private static int sx_offset = 2;
    private static int sy_offset = 3;
    private static int acc_x_offset = 4;
    private static int acc_y_offset = 5;
    private static int bx_offset = 6;
    private static int by_offset = 7;
    private static int bw_offset = 8;
    private static int bh_offset = 9;
    private static int sp_offset = 10;
    private static int ep_offset = 11;
    private static int se_offset = 12;
    private static int ee_offset = 13;

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

    public float bounds_x()
    {
        return Main.Memory.body_buffer[index() + bx_offset];
    }

    public float bounds_y()
    {
        return Main.Memory.body_buffer[index() + by_offset];
    }

    public float bounds_w()
    {
        return Main.Memory.body_buffer[index() + bw_offset];
    }

    public float bounds_h()
    {
        return Main.Memory.body_buffer[index() + bh_offset];
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
